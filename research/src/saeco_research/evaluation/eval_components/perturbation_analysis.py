from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Literal

import torch
import tqdm
from pydantic import BaseModel
from saeco_research.evaluation.cache_version import cache_version
from saeco_research.evaluation.storage.chunk import Chunk
from torch import Tensor

if TYPE_CHECKING:
    from saeco_research.evaluation.evaluation import Evaluation


Pooling = Literal["max", "mean", "sum", "count", "any"]
DoseMode = Literal["max", "slope"]
ProfileAggregation = Literal["mean", "median"]
SimilarityMode = Literal["profile", "pattern"]


class PerturbationConfig(BaseModel, frozen=True):
    """
    Session-wide configuration for perturbation analysis.
    """

    # Control condition
    control_drug: str = "DMSO_TF"
    control_dosage: int = 0

    # Metadata keys
    cell_line_key: str = "cell_line"
    drug_key: str = "drug"
    dosage_key: str = "dosage"

    # Analysis defaults
    doses: tuple[int, ...] = (1, 2, 3)
    pooling: Pooling = "max"
    dose_mode: DoseMode = "max"
    aggregation: ProfileAggregation = "mean"
    normalize_across_cell_lines: bool = True


# ---------------------------------------------------------------------
# Small internal primitives (keep methods short + reusable)
# ---------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class _RootMeta:
    drug: Tensor
    cell_line: Tensor
    dosage: Tensor

    def slice_for_docs(
        self, doc_ids_cpu: Tensor, *, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor]:
        doc_ids_cpu = doc_ids_cpu.to(dtype=torch.long, device="cpu")
        return (
            self.drug[doc_ids_cpu].to(device),
            self.cell_line[doc_ids_cpu].to(device),
            self.dosage[doc_ids_cpu].to(device),
        )


@dataclass(slots=True, frozen=True)
class _ChunkBatch:
    doc_ids_cpu: Tensor
    agg_acts: Tensor  # (n_docs, d_dict) float32 on device
    drug_tokens: Tensor
    cell_tokens: Tensor
    dose_values: Tensor


@dataclass(slots=True, frozen=True)
class _SortedMetadataTokenMap:
    """
    Vectorized token -> value mapping without assuming tokens are small/contiguous.

    Uses searchsorted on sorted unique keys.
    """

    keys_sorted: Tensor  # (k,) long, sorted
    sorted_to_value: Tensor  # (k,) long, value for each sorted position

    @classmethod
    def from_pairs(
        cls, keys: Sequence[int], values: Sequence[int], *, device: torch.device
    ) -> _SortedMetadataTokenMap:
        keys_t = torch.tensor(list(keys), dtype=torch.long)
        vals_t = torch.tensor(list(values), dtype=torch.long)

        if keys_t.numel() != keys_t.unique().numel():
            raise ValueError(
                "Duplicate tokens in mapping keys (ambiguous token->row mapping)."
            )

        order = keys_t.argsort()
        return cls(
            keys_sorted=keys_t[order].to(device),
            sorted_to_value=vals_t[order].to(device),
        )

    def lookup(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        k = int(self.keys_sorted.numel())
        if k == 0:
            out = torch.full_like(tokens, -1)
            return out, torch.zeros_like(tokens, dtype=torch.bool)

        pos = torch.searchsorted(self.keys_sorted, tokens)
        in_bounds = pos < k
        pos_safe = pos.clamp(max=k - 1)

        matched = in_bounds & (self.keys_sorted[pos_safe] == tokens)
        out = torch.full_like(tokens, -1)
        out[matched] = self.sorted_to_value[pos_safe[matched]]
        return out, matched


@dataclass(slots=True, frozen=True)
class _DoseIndexer:
    """
    Maps raw dosage values -> dose_idx (in cfg.doses ordering), with optional exclusion of control dosage.
    """

    map_: _SortedMetadataTokenMap
    n_doses: int
    control_dosage: int | None

    @classmethod
    def from_cfg(cls, cfg: PerturbationConfig, *, device: torch.device) -> _DoseIndexer:
        doses = list(cfg.doses)
        return cls(
            map_=_SortedMetadataTokenMap.from_pairs(
                keys=doses,
                values=list(range(len(doses))),
                device=device,
            ),
            n_doses=len(doses),
            control_dosage=int(cfg.control_dosage)
            if cfg.control_dosage is not None
            else None,
        )

    def index(self, dose_values: Tensor) -> tuple[Tensor, Tensor]:
        dose_idx, ok = self.map_.lookup(dose_values.to(torch.long))
        if self.control_dosage is None:
            return dose_idx, ok

        not_control = dose_values.to(torch.long) != int(self.control_dosage)
        ok = ok & not_control
        return dose_idx.masked_fill(~ok, -1), ok


@dataclass(slots=True)
class _GroupAccumulator:
    sums: Tensor  # (n_groups, d_dict) float32
    counts: Tensor  # (n_groups,) long

    @classmethod
    def zeros(
        cls, *, n_groups: int, d_dict: int, device: torch.device
    ) -> _GroupAccumulator:
        return cls(
            sums=torch.zeros((n_groups, d_dict), dtype=torch.float32, device=device),
            counts=torch.zeros((n_groups,), dtype=torch.long, device=device),
        )

    def add(self, group_ids: Tensor, values: Tensor) -> None:
        if group_ids.numel() == 0:
            return
        self.sums.index_add_(0, group_ids.to(torch.long), values.to(torch.float32))
        self.counts += torch.bincount(
            group_ids.to(torch.long), minlength=int(self.counts.numel())
        )

    def means(self) -> Tensor:
        denom = self.counts.clamp(min=1).to(torch.float32).unsqueeze(1)
        out = self.sums / denom
        return torch.where(
            self.counts.unsqueeze(1) > 0,
            out,
            torch.full_like(out, torch.nan),
        )


# ---------------------------------------------------------------------
# Main mixin
# ---------------------------------------------------------------------


class PerturbationAnalysis:
    """
    Utilities for perturbation analysis:
      - drug deltas vs control by (cell_line, dose)
      - drug profiles + similarity
      - sensitivity + correlation against control features

    IMPORTANT:
      This version does NOT treat metadata tokens as array indices.
      Tokens are only used for equality checks and for token->row mapping.
    """

    # ---------------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------------

    @property
    def perturbation_config(self: Evaluation) -> PerturbationConfig:
        cfg = getattr(self, "_perturbation_config", None)
        return cfg if cfg is not None else PerturbationConfig()

    @perturbation_config.setter
    def perturbation_config(self: Evaluation, cfg: PerturbationConfig) -> None:
        self._perturbation_config = cfg

    # ---------------------------------------------------------------------
    # Metadata helpers
    # ---------------------------------------------------------------------

    def _meta_root(self: Evaluation) -> Evaluation:
        return self.root

    def _root_metadata_tensor(self: Evaluation, key: str) -> Tensor:
        meta = self._meta_root().metadata_store[key]
        assert isinstance(meta, Tensor), f"Metadata {key} must be a torch.Tensor"
        return meta

    def _root_meta(self: Evaluation, cfg: PerturbationConfig) -> _RootMeta:
        return _RootMeta(
            drug=self._root_metadata_tensor(cfg.drug_key),
            cell_line=self._root_metadata_tensor(cfg.cell_line_key),
            dosage=self._root_metadata_tensor(cfg.dosage_key),
        )

    def _md_info(self: Evaluation, key: str):
        md = self._meta_root().metadata_store.get(key)
        if md is None:
            raise KeyError(f"Missing metadata key: {key!r}")
        info = getattr(md, "info", None)
        return info

    def get_metadata_id(self: Evaluation, key: str, value: str | int) -> int:
        if isinstance(value, int):
            return value
        # handy fallback when we used numeric-string IDs without translators
        if value.isdigit():
            return int(value)

        info = self._md_info(key)
        if info is None or info.fromstr is None:
            raise ValueError(
                f"Metadata '{key}' does not have a string->id translator (info.fromstr is None)."
            )
        try:
            return int(info.fromstr[value])
        except KeyError as e:
            raise KeyError(f"Unknown metadata value for {key=}: {value!r}") from e

    def get_metadata_str(self: Evaluation, key: str, value_id: int) -> str:
        info = self._md_info(key)
        if info is None or info.tostr is None:
            return str(int(value_id))
        try:
            return str(info.tostr[int(value_id)])
        except KeyError as e:
            raise KeyError(f"Unknown metadata id for {key=}: {value_id}") from e

    def get_all_metadata_strings(
        self: Evaluation,
        key: str,
        *,
        exclude_special: bool = True,
        special_prefix: str = "<<",
    ) -> list[str]:
        info = self._md_info(key)
        if info is None or info.tostr is None:
            ids = self._root_metadata_tensor(key).unique(sorted=True).tolist()
            return [str(int(i)) for i in ids]

        out = [info.tostr[i] for i in sorted(info.tostr.keys())]
        if exclude_special:
            out = [s for s in out if not s.startswith(special_prefix)]
        return out

    def get_all_cell_lines(
        self: Evaluation,
        *,
        exclude_special: bool = True,
        config: PerturbationConfig | None = None,
    ) -> list[str]:
        cfg = config or self.perturbation_config
        return self.get_all_metadata_strings(
            cfg.cell_line_key, exclude_special=exclude_special
        )

    def get_all_drugs(
        self: Evaluation,
        *,
        exclude_special: bool = True,
        exclude_control: bool = True,
        config: PerturbationConfig | None = None,
    ) -> list[str]:
        cfg = config or self.perturbation_config
        drugs = self.get_all_metadata_strings(
            cfg.drug_key, exclude_special=exclude_special
        )
        return [d for d in drugs if (not exclude_control) or d != cfg.control_drug]

    # ---------------------------------------------------------------------
    # Chunk iteration + seq aggregation
    # ---------------------------------------------------------------------

    def _chunk_doc_ids(self: Evaluation, chunk: Chunk) -> Tensor:
        cfg = chunk.cfg
        start = cfg.docs_per_chunk * chunk.idx
        stop = cfg.docs_per_chunk * (chunk.idx + 1)

        doc_ids = torch.arange(start, stop, dtype=torch.long)
        nf = getattr(chunk, "named_filter", None)
        if nf is None or nf.filter is None:
            return doc_ids

        mask = nf.filter[start:stop].to(torch.bool).cpu()
        return doc_ids[mask]

    def _seq_aggregate_dense(
        self: Evaluation, acts: Tensor, pooling: Pooling
    ) -> Tensor:
        if pooling == "max":
            return acts.max(dim=1).values
        if pooling == "mean":
            return acts.mean(dim=1)
        if pooling == "sum":
            return acts.sum(dim=1)
        if pooling == "count":
            return (acts > 0).sum(dim=1).to(dtype=acts.dtype)
        if pooling == "any":
            return (acts > 0).any(dim=1).to(dtype=acts.dtype)
        raise ValueError(f"Unknown pooling={pooling!r}")

    def iter_seq_aggregated_acts_by_chunk(
        self: Evaluation,
        *,
        pooling: Pooling = "max",
        device: torch.device | str | None = None,
        show_progress: bool = True,
    ) -> Iterable[tuple[Tensor, Tensor]]:
        """
        Yields:
          doc_ids_cpu: (n_docs,)
          agg_acts: (n_docs, d_dict) float32 on device
        """
        dev = torch.device(self.device) if device is None else torch.device(device)
        chunks = self.cached_acts.chunks

        for chunk in tqdm.tqdm(chunks, total=len(chunks), disable=not show_progress):
            doc_ids_cpu = self._chunk_doc_ids(chunk)
            acts = chunk.acts.value
            assert isinstance(acts, Tensor), "Expected chunk.acts.value to be a Tensor"

            acts = acts.to(dev)
            if acts.is_sparse:
                acts = acts.coalesce().to_dense()

            agg = self._seq_aggregate_dense(acts, pooling=pooling).to(
                dtype=torch.float32
            )
            yield doc_ids_cpu, agg

    def iter_chunk_batches(
        self: Evaluation,
        *,
        config: PerturbationConfig | None = None,
        device: torch.device | str | None = None,
        show_progress: bool = True,
    ) -> Iterable[_ChunkBatch]:
        cfg = config or self.perturbation_config
        dev = torch.device(self.device) if device is None else torch.device(device)
        root_meta = self._root_meta(cfg)

        for doc_ids_cpu, agg in self.iter_seq_aggregated_acts_by_chunk(
            pooling=cfg.pooling,
            device=dev,
            show_progress=show_progress,
        ):
            drug_tokens, cell_tokens, dose_values = root_meta.slice_for_docs(
                doc_ids_cpu, device=dev
            )
            yield _ChunkBatch(
                doc_ids_cpu=doc_ids_cpu,
                agg_acts=agg,
                drug_tokens=drug_tokens,
                cell_tokens=cell_tokens,
                dose_values=dose_values,
            )

    # ---------------------------------------------------------------------
    # Mapping helpers (NO token-as-index behavior)
    # ---------------------------------------------------------------------

    def _cell_line_row_map(
        self: Evaluation,
        cell_lines: Sequence[str],
        *,
        cfg: PerturbationConfig,
        device: torch.device,
    ) -> _SortedMetadataTokenMap:
        cell_tokens = [self.get_metadata_id(cfg.cell_line_key, cl) for cl in cell_lines]
        row_ids = list(range(len(cell_lines)))
        return _SortedMetadataTokenMap.from_pairs(cell_tokens, row_ids, device=device)

    def _resolve_cell_lines(
        self: Evaluation,
        cell_lines: Sequence[str] | None,
        *,
        cfg: PerturbationConfig,
    ) -> list[str]:
        if cell_lines is None:
            return self.get_all_cell_lines(exclude_special=True, config=cfg)
        return list(cell_lines)

    # ---------------------------------------------------------------------
    # Control stats
    # ---------------------------------------------------------------------

    @staticmethod
    def _pack_sums_counts(sums_cpu: Tensor, counts_cpu: Tensor) -> Tensor:
        # packed: (n, d_dict + 1) where last column is counts (float32)
        return torch.cat(
            [sums_cpu.to(torch.float32), counts_cpu.to(torch.float32).unsqueeze(1)],
            dim=1,
        )

    @staticmethod
    def _unpack_sums_counts(packed: Tensor) -> tuple[Tensor, Tensor]:
        if packed.ndim != 2 or packed.shape[1] < 2:
            raise ValueError(f"Unexpected packed shape: {tuple(packed.shape)}")
        return packed[:, :-1], packed[:, -1].to(torch.long)

    @cache_version(2)
    @torch.inference_mode()
    def control_sums_counts_by_cell_line(
        self: Evaluation,
        *,
        cell_lines: Sequence[str] | None = None,
        config: PerturbationConfig | None = None,
        show_progress: bool = True,
    ) -> Tensor:
        """
        Returns packed (n_cell_lines, d_dict+1) on CPU:
          packed[:, :-1] = sums
          packed[:, -1]  = counts
        Order is EXACTLY the provided `cell_lines` order (or the default list).
        """
        cfg = config or self.perturbation_config
        chosen_cell_lines = self._resolve_cell_lines(cell_lines, cfg=cfg)

        dev = torch.device(self.device)
        cell_row = self._cell_line_row_map(chosen_cell_lines, cfg=cfg, device=dev)
        control_drug_token = self.get_metadata_id(cfg.drug_key, cfg.control_drug)

        acc = _GroupAccumulator.zeros(
            n_groups=len(chosen_cell_lines), d_dict=self.d_dict, device=dev
        )

        for batch in self.iter_chunk_batches(
            config=cfg, device=dev, show_progress=show_progress
        ):
            is_control = (batch.drug_tokens == control_drug_token) & (
                batch.dose_values == int(cfg.control_dosage)
            )
            if not is_control.any():
                continue

            rows, ok = cell_row.lookup(batch.cell_tokens)
            keep = is_control & ok
            if not keep.any():
                continue

            acc.add(rows[keep], batch.agg_acts[keep])

        sums_cpu = acc.sums.detach().cpu()
        counts_cpu = acc.counts.detach().cpu()
        return self._pack_sums_counts(sums_cpu, counts_cpu)

    def control_means_by_cell_line(
        self: Evaluation,
        *,
        cell_lines: Sequence[str] | None = None,
        config: PerturbationConfig | None = None,
        show_progress: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """
        Returns (means_cpu, counts_cpu) where means are aligned to `cell_lines` order.
        """
        cfg = config or self.perturbation_config
        chosen_cell_lines = self._resolve_cell_lines(cell_lines, cfg=cfg)

        packed = self.cached.control_sums_counts_by_cell_line(
            cell_lines=chosen_cell_lines,
            config=cfg,
            show_progress=show_progress,
        )
        sums, counts = self._unpack_sums_counts(packed)

        denom = counts.clamp(min=1).to(torch.float32).unsqueeze(1)
        means = sums / denom
        means = torch.where(
            counts.unsqueeze(1) > 0, means, torch.full_like(means, torch.nan)
        )
        return means, counts

    # ---------------------------------------------------------------------
    # Drug deltas (core primitive)
    # ---------------------------------------------------------------------

    @cache_version(2)
    @torch.inference_mode()
    def compute_drug_deltas_tensor(
        self: Evaluation,
        drug: str,
        cell_lines: Sequence[str] | None = None,
        *,
        config: PerturbationConfig | None = None,
        show_progress: bool = True,
    ) -> Tensor:
        """
        Returns (n_cell_lines, n_doses, d_dict) on CPU, aligned to `cell_lines` order.
        """
        cfg = config or self.perturbation_config
        chosen_cell_lines = self._resolve_cell_lines(cell_lines, cfg=cfg)

        dev = torch.device(self.device)
        cell_row = self._cell_line_row_map(chosen_cell_lines, cfg=cfg, device=dev)
        dose_indexer = _DoseIndexer.from_cfg(cfg, device=dev)

        drug_token = self.get_metadata_id(cfg.drug_key, drug)
        control_means_cpu, _ = self.control_means_by_cell_line(
            cell_lines=chosen_cell_lines,
            config=cfg,
            show_progress=show_progress,
        )
        control_means = control_means_cpu.to(dev)

        acc = _GroupAccumulator.zeros(
            n_groups=len(chosen_cell_lines) * dose_indexer.n_doses,
            d_dict=self.d_dict,
            device=dev,
        )

        for batch in self.iter_chunk_batches(
            config=cfg, device=dev, show_progress=show_progress
        ):
            dose_idx, dose_ok = dose_indexer.index(batch.dose_values)
            rows, row_ok = cell_row.lookup(batch.cell_tokens)

            keep = (batch.drug_tokens == drug_token) & dose_ok & row_ok
            if not keep.any():
                continue

            group = rows[keep] * dose_indexer.n_doses + dose_idx[keep]
            acc.add(group, batch.agg_acts[keep])

        means = acc.means().view(
            len(chosen_cell_lines), dose_indexer.n_doses, self.d_dict
        )
        deltas = means - control_means.unsqueeze(1)
        return deltas.detach().cpu()

    @cache_version(2)
    def compute_drug_deltas_matrix(
        self: Evaluation,
        drug: str,
        cell_lines: Sequence[str] | None = None,
        *,
        config: PerturbationConfig | None = None,
        show_progress: bool = True,
    ) -> Tensor:
        """
        Returns (n_cell_lines, d_dict) on CPU, aligned to `cell_lines` order.
        """
        cfg = config or self.perturbation_config
        deltas_3d = self.cached.compute_drug_deltas_tensor(
            drug=drug,
            cell_lines=cell_lines,
            config=cfg,
            show_progress=show_progress,
        ).to(torch.float32)

        if cfg.dose_mode == "max":
            return deltas_3d[:, -1, :]

        if cfg.dose_mode == "slope":
            x = torch.tensor(list(cfg.doses), dtype=torch.float32)
            x = x - x.mean()
            denom = (x * x).sum().clamp(min=1e-8)

            y = torch.nan_to_num(deltas_3d, nan=0.0)
            return (y * x.view(1, -1, 1)).sum(dim=1) / denom

        raise ValueError(f"Unknown dose_mode={cfg.dose_mode!r}")

    # ---------------------------------------------------------------------
    # Drug profiles + similarity
    # ---------------------------------------------------------------------

    @cache_version(2)
    def compute_drug_profile(
        self: Evaluation,
        drug: str,
        cell_lines: Sequence[str] | None = None,
        *,
        config: PerturbationConfig | None = None,
        show_progress: bool = True,
    ) -> Tensor:
        """
        Returns (d_dict,) profile aggregated across cell lines.
        """
        cfg = config or self.perturbation_config
        matrix = self.cached.compute_drug_deltas_matrix(
            drug=drug,
            cell_lines=cell_lines,
            config=cfg,
            show_progress=show_progress,
        ).to(torch.float32)

        x = torch.nan_to_num(matrix, nan=0.0)
        if cfg.normalize_across_cell_lines:
            mu = x.mean(dim=0)
            sd = x.std(dim=0, unbiased=False).clamp(min=1e-8)
            x = (x - mu) / sd

        if cfg.aggregation == "mean":
            return torch.nan_to_num(x.mean(dim=0), nan=0.0)
        if cfg.aggregation == "median":
            return torch.nan_to_num(x.median(dim=0).values, nan=0.0)

        raise ValueError(f"Unknown aggregation={cfg.aggregation!r}")

    @staticmethod
    def _cosine_similarity_matrix(vectors: Tensor, *, eps: float = 1e-8) -> Tensor:
        v = vectors.to(torch.float32)
        n = v / v.norm(dim=1, keepdim=True).clamp(min=eps)
        return (n @ n.T).to(torch.float32)

    def _similarity_vectors(
        self: Evaluation,
        drugs: Sequence[str],
        cell_lines: Sequence[str] | None,
        *,
        mode: SimilarityMode,
        cfg: PerturbationConfig,
        show_progress: bool,
    ) -> Tensor:
        it = tqdm.tqdm(drugs, disable=not show_progress)

        if mode == "profile":
            vecs = [
                self.cached.compute_drug_profile(
                    drug=d,
                    cell_lines=cell_lines,
                    config=cfg,
                    show_progress=False,
                )
                for d in it
            ]
            return torch.stack(vecs).to(torch.float32)

        if mode == "pattern":
            vecs = [
                torch.nan_to_num(
                    self.cached.compute_drug_deltas_matrix(
                        drug=d,
                        cell_lines=cell_lines,
                        config=cfg,
                        show_progress=False,
                    ).to(torch.float32),
                    nan=0.0,
                ).flatten()
                for d in it
            ]
            return torch.stack(vecs).to(torch.float32)

        raise ValueError(f"Unknown mode={mode!r}")

    # @cache_version(2)
    # def compute_drug_similarity_matrix(
    #     self: Evaluation,
    #     drugs: list[str] | None = None,
    #     cell_lines: Sequence[str] | None = None,
    #     *,
    #     mode: SimilarityMode = "profile",
    #     config: PerturbationConfig | None = None,
    #     show_progress: bool = True,
    # ) -> tuple[Tensor, list[str]]:
    #     cfg = config or self.perturbation_config
    #     chosen_drugs = drugs or self.get_all_drugs(
    #         exclude_special=True, exclude_control=True, config=cfg
    #     )

    #     vectors = self._similarity_vectors(
    #         chosen_drugs,
    #         cell_lines,
    #         mode=mode,
    #         cfg=cfg,
    #         show_progress=show_progress,
    #     )
    #     return self._cosine_similarity_matrix(vectors), chosen_drugs

    def iter_chunk_and_agg(
        self: Evaluation,
        *,
        pooling: Pooling = "max",
        device: torch.device | str | None = None,
        show_progress: bool = True,
    ) -> Iterable[tuple[Chunk, Tensor]]:
        device = self.device if device is None else torch.device(device)

        chunks = self.cached_acts.chunks
        it = tqdm.tqdm(chunks, total=len(chunks), disable=not show_progress)
        for chunk in it:
            acts_ft = chunk.acts
            v = acts_ft.value
            assert isinstance(v, Tensor), "Expected chunk.acts.value to be a Tensor"
            v = v.to(device)
            if v.is_sparse:
                v = v.coalesce().to_dense()
            # v: (doc, seq, d_dict)
            agg = self._seq_aggregate_dense(v, pooling=pooling).to(dtype=torch.float32)
            yield chunk, agg

    @cache_version(1)
    def compute_metadata_effect_profile(
        self: Evaluation,
        metadata_map: MetadataValueMap,
        pooling: Pooling = "max",
    ):
        """
        Returns (n_drugs, n_drugs) cosine similarity.

        mode:
          - "profile": cosine similarity over (d_dict,) profiles
          - "pattern": cosine similarity over flattened (n_cell_lines, d_dict) deltas

        Warning:
          pattern mode can be very large if you pass many drugs and many cell lines.
        """
        metadata_effects = torch.zeros(
            len(metadata_map.value_strings),
            self.d_dict,
            dtype=torch.float32,
            device=self.device,
        )
        metadata = metadata_map.get_metadata(self)
        num_matched = torch.zeros(
            len(metadata_map.value_strings),
            dtype=torch.float32,
            device=self.device,
        )
        for chunk, agg in self.iter_chunk_and_agg(pooling=pooling):
            chunk_metadata = metadata[chunk.doc_ids]
            for i in range(len(metadata_map.value_strings)):  # TODO scatter add this
                metadata_effects[i] += (chunk_metadata == metadata_map.ids[i]).to(
                    agg.device, torch.float32
                ) @ agg
                num_matched[i] += (chunk_metadata == metadata_map.ids[i]).sum()
        return metadata_effects / num_matched.unsqueeze(1)

    def top_similar_drugs(
        self: Evaluation,
        sim_matrix: Tensor,
        drugs: Sequence[str],
        query: str,
        *,
        k: int = 10,
    ) -> list[tuple[str, float]]:
        idx = drugs.index(query)
        sims = sim_matrix[idx].clone()
        sims[idx] = -torch.inf

        top = sims.topk(min(k, sims.numel() - 1), largest=True)
        return [
            (drugs[int(i)], float(v))
            for i, v in zip(top.indices, top.values, strict=True)
        ]

    def top_shared_differential_features(
        self: Evaluation,
        drug1_profile: Tensor,
        drug2_profile: Tensor,
        *,
        k: int = 10,
    ) -> list[tuple[int, float, float, float]]:
        p1 = drug1_profile.to(torch.float32)
        p2 = drug2_profile.to(torch.float32)

        contrib = p1 * p2
        top = contrib.topk(min(k, contrib.numel()), largest=True)

        return [
            (int(fid), float(p1[fid].item()), float(p2[fid].item()), float(c))
            for fid, c in zip(top.indices.tolist(), top.values.tolist(), strict=True)
        ]

    # ---------------------------------------------------------------------
    # Cytotox helpers (thin wrappers)
    # ---------------------------------------------------------------------

    def validate_cytotox_feature(
        self: Evaluation,
        feature_id: int,
        *,
        token_k: int = 500,
        token_mode: str = "active",
        patch_subset_n: int = 500,
    ):
        token_enrich = self.top_activations_token_enrichments(
            feature=feature_id,
            k=token_k,
            mode=token_mode,
        )
        logit_effects = self.average_aggregated_patching_effect_on_dataset(
            feature_id=feature_id,
            random_subset_n=patch_subset_n,
        )
        return token_enrich, logit_effects

    # ---------------------------------------------------------------------
    # Act 2: sensitivity + correlation
    # ---------------------------------------------------------------------

    def compute_sensitivity_by_cell_line(
        self: Evaluation,
        drug: str,
        cell_lines: Sequence[str] | None,
        *,
        response_feature: int,
        config: PerturbationConfig | None = None,
        show_progress: bool = True,
    ) -> dict[str, float]:
        cfg = config or self.perturbation_config
        chosen_cell_lines = self._resolve_cell_lines(cell_lines, cfg=cfg)

        deltas = self.cached.compute_drug_deltas_matrix(
            drug=drug,
            cell_lines=chosen_cell_lines,
            config=cfg,
            show_progress=show_progress,
        )
        sens = deltas[:, int(response_feature)]
        return {cl: float(sens[i].item()) for i, cl in enumerate(chosen_cell_lines)}

    def compute_control_features_by_cell_line(
        self: Evaluation,
        cell_lines: Sequence[str] | None,
        *,
        config: PerturbationConfig | None = None,
        show_progress: bool = True,
    ) -> Tensor:
        cfg = config or self.perturbation_config
        chosen_cell_lines = self._resolve_cell_lines(cell_lines, cfg=cfg)
        means, _ = self.control_means_by_cell_line(
            cell_lines=chosen_cell_lines,
            config=cfg,
            show_progress=show_progress,
        )
        return means

    @staticmethod
    def _pearson_r_features(X: Tensor, y: Tensor, *, eps: float) -> Tensor:
        y = y.to(torch.float32)
        X = X.to(torch.float32)

        yc = y - y.mean()
        Xc = X - X.mean(dim=0)

        num = (Xc * yc.unsqueeze(1)).sum(dim=0)
        den = (Xc.pow(2).sum(dim=0).sqrt() * yc.pow(2).sum().sqrt()).clamp(min=eps)
        return num / den

    @cache_version(2)
    def compute_feature_sensitivity_correlation(
        self: Evaluation,
        drug: str,
        cell_lines: Sequence[str] | None,
        *,
        response_feature: int,
        config: PerturbationConfig | None = None,
        show_progress: bool = True,
        eps: float = 1e-8,
    ) -> Tensor:
        cfg = config or self.perturbation_config
        chosen_cell_lines = self._resolve_cell_lines(cell_lines, cfg=cfg)

        deltas = self.cached.compute_drug_deltas_matrix(
            drug=drug,
            cell_lines=chosen_cell_lines,
            config=cfg,
            show_progress=show_progress,
        ).to(torch.float32)

        sens = deltas[:, int(response_feature)]
        control_feats = self.compute_control_features_by_cell_line(
            cell_lines=chosen_cell_lines,
            config=cfg,
            show_progress=show_progress,
        ).to(torch.float32)

        valid = torch.isfinite(sens)
        sens = torch.nan_to_num(sens[valid], nan=0.0)
        control_feats = torch.nan_to_num(control_feats[valid], nan=0.0)

        return self._pearson_r_features(control_feats, sens, eps=eps)

    def get_metadata_values_and_strings(
        self: Evaluation,
        key: str,
        *,
        exclude_special: bool = True,
        special_prefix: str = "<<",
    ) -> MetadataValueMap:
        md = self._meta_root().metadata_store.get(key)
        info = md.info
        if info is None or info.tostr is None or info.fromstr is None:
            # Fallback: infer from tensor unique values (no translator)
            raise ValueError("Metadata has no translator")

        out = [info.tostr[i] for i in sorted(info.tostr.keys())]
        if exclude_special:
            out = [s for s in out if not s.startswith(special_prefix)]
        return MetadataValueMap(
            metadata_key=key,
            ids=tuple([info.fromstr[s] for s in out]),
            value_strings=tuple(out),
        )


class MetadataValueMap(BaseModel):
    metadata_key: str
    ids: tuple[int, ...]
    value_strings: tuple[str, ...]

    @cached_property
    def indices(self) -> Tensor:
        return torch.arange(len(self.ids))

    def __len__(self) -> int:
        return len(self.ids)

    def get_metadata(self, e: Evaluation) -> Tensor:
        return e.metadata_store[self.metadata_key]
