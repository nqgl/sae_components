from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Literal

import torch
import tqdm
from pydantic import BaseModel
from torch import Tensor

from saeco.evaluation.cache_version import cache_version
from saeco.evaluation.storage.chunk import Chunk

if TYPE_CHECKING:
    from saeco.evaluation.evaluation import Evaluation


Pooling = Literal["max", "mean", "sum", "count", "any"]
DoseMode = Literal["max", "slope"]
ProfileAggregation = Literal["mean", "median"]
SimilarityMode = Literal["profile", "pattern"]


class PerturbationConfig(BaseModel, frozen=True):
    """
    Session-wide configuration for perturbation analysis.

    This config controls default values for metadata keys, control conditions,
    and analysis parameters. Individual method calls can override by passing
    a different config.
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


class PerturbationAnalysis:
    """
    Act 1 + Act 2 utilities:
      - Compute drug deltas vs control per (cell_line, dose)
      - Build drug profiles and drug-drug similarity
      - Compute cell-line sensitivity signals and correlate control features to sensitivity

    Notes on indexing:
      - This code assumes doc indices are GLOBAL (0..root.num_samples-1), matching how your
        Chunk filtering/FilteredTensor virtual shapes are set up.
      - It works on both root and filtered evals as long as metadatas exist on root.

    Configuration:
      - Use `perturbation_config` property to set session-wide defaults for control drug,
        metadata keys, doses, pooling, etc.
      - Individual methods accept `config: PerturbationConfig | None` to override defaults.
    """

    # ---------------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------------

    @property
    def perturbation_config(self: Evaluation) -> PerturbationConfig:
        """Get the current perturbation analysis configuration."""
        cfg = getattr(self, "_perturbation_config", None)
        if cfg is None:
            return PerturbationConfig()
        return cfg

    @perturbation_config.setter
    def perturbation_config(self: Evaluation, cfg: PerturbationConfig) -> None:
        """Set the perturbation analysis configuration."""
        self._perturbation_config = cfg

    # ---------------------------------------------------------------------
    # Metadata helpers
    # ---------------------------------------------------------------------

    def _meta_root(self: Evaluation) -> Evaluation:
        return self.root  # unfiltered root evaluation

    def _root_metadata_tensor(self: Evaluation, key: str) -> Tensor:
        meta = self._meta_root().metadata_store[key]
        assert isinstance(meta, Tensor), f"Metadata {key} must be a torch.Tensor"
        return meta

    def get_metadata_id(self: Evaluation, key: str, value: str | int) -> int:
        """
        Translate string metadata value -> integer ID (if already int, passthrough).
        Requires str translator to exist for this metadata.
        """
        if isinstance(value, int):
            return value
        md = self._meta_root().metadata_store.get(key)
        info = md.info
        if info is None or info.fromstr is None:
            raise ValueError(
                f"Metadata '{key}' does not have a string->id translator (info.fromstr is None). "
                f"Did you call metadatas.set_str_translator('{key}', ...)?"
            )
        try:
            return int(info.fromstr[value])
        except KeyError as e:
            raise KeyError(f"Unknown metadata value for {key=}: {value!r}") from e

    def get_metadata_str(self: Evaluation, key: str, value_id: int) -> str:
        """
        Translate integer ID -> string metadata value.
        """
        md = self._meta_root().metadata_store.get(key)
        info = md.info
        if info is None or info.tostr is None:
            raise ValueError(
                f"Metadata '{key}' does not have an id->string translator (info.tostr is None)."
            )
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
        """
        List all known string labels for a metadata key, using its translator.
        """
        md = self._meta_root().metadata_store.get(key)
        info = md.info
        if info is None or info.tostr is None:
            # Fallback: infer from tensor unique values (no translator)
            meta = self._root_metadata_tensor(key)
            ids = meta.unique(sorted=True).tolist()
            return [str(i) for i in ids]

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
        if exclude_control:
            drugs = [d for d in drugs if d != cfg.control_drug]
        return drugs

    # ---------------------------------------------------------------------
    # Chunk iteration + seq-aggregation
    # ---------------------------------------------------------------------

    def _chunk_doc_ids(self: Evaluation, chunk: Chunk) -> Tensor:
        """
        Return the *global* doc indices corresponding to the docs present in this chunk's
        loaded tensors (after any NamedFilter masking).
        """
        cfg = chunk.cfg
        start = cfg.docs_per_chunk * chunk.idx
        stop = cfg.docs_per_chunk * (chunk.idx + 1)
        doc_ids = torch.arange(start, stop, dtype=torch.long)

        nf = getattr(chunk, "named_filter", None)
        if nf is not None and nf.filter is not None:
            mask = nf.filter[start:stop].to(torch.bool).cpu()
            doc_ids = doc_ids[mask]
        return doc_ids

    def _seq_aggregate_dense(
        self: Evaluation, acts: Tensor, pooling: Pooling
    ) -> Tensor:
        """
        acts: (doc, seq, d_dict) dense
        returns: (doc, d_dict)
        """
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
        Yields (doc_ids, agg_acts) per chunk.

        doc_ids: (n_docs_in_chunk,)
        agg_acts: (n_docs_in_chunk, d_dict)
        """
        device = self.device if device is None else torch.device(device)

        chunks = self.cached_acts.chunks
        it = tqdm.tqdm(chunks, total=len(chunks), disable=not show_progress)
        for chunk in it:
            doc_ids = self._chunk_doc_ids(chunk)  # CPU
            acts_ft = chunk.acts
            v = acts_ft.value
            assert isinstance(v, Tensor), "Expected chunk.acts.value to be a Tensor"
            v = v.to(device)
            if v.is_sparse:
                v = v.coalesce().to_dense()
            # v: (doc, seq, d_dict)
            agg = self._seq_aggregate_dense(v, pooling=pooling).to(dtype=torch.float32)
            yield doc_ids, agg

    # ---------------------------------------------------------------------
    # Control stats
    # ---------------------------------------------------------------------

    @cache_version(1)
    @torch.inference_mode()
    def control_sums_counts_by_cell_line(
        self: Evaluation,
        *,
        config: PerturbationConfig | None = None,
        show_progress: bool = True,
    ) -> Tensor:
        """
        Returns a packed tensor holding:
          - sums:   (n_cell_lines, d_dict)
          - counts: (n_cell_lines,)

        Packed as a single tensor for caching convenience:
          out[0] = sums (float32)
          out[1, :, 0] = counts (float32)  # counts stored in a minimal shape slice

        Consumers should use `unpack_control_sums_counts(...)`.
        """
        cfg = config or self.perturbation_config
        root = self._meta_root()
        drug_meta = root.metadata_store[cfg.drug_key]
        cell_meta = root.metadata_store[cfg.cell_line_key]
        dose_meta = root.metadata_store[cfg.dosage_key]
        assert (
            isinstance(drug_meta, Tensor)
            and isinstance(cell_meta, Tensor)
            and isinstance(dose_meta, Tensor)
        )

        control_drug_id = self.get_metadata_id(cfg.drug_key, cfg.control_drug)

        n_cell_lines = int(cell_meta.max().item()) + 1
        d_dict = self.d_dict

        device = self.device
        sums = torch.zeros((n_cell_lines, d_dict), dtype=torch.float32, device=device)
        counts = torch.zeros((n_cell_lines,), dtype=torch.long, device=device)

        for doc_ids, agg in self.iter_seq_aggregated_acts_by_chunk(
            pooling=cfg.pooling, device=device, show_progress=show_progress
        ):
            # Pull metadata for these docs
            did = doc_ids.to(torch.long)
            d = drug_meta[did].to(device)
            c = cell_meta[did].to(device)
            z = dose_meta[did].to(device)

            m = (d == control_drug_id) & (z == cfg.control_dosage)
            if not m.any():
                continue

            c_sel = c[m]
            a_sel = agg[m]  # already float32 on device

            sums.index_add_(0, c_sel, a_sel)
            counts += torch.bincount(c_sel, minlength=n_cell_lines)

        # Pack for caching (store on CPU)
        sums_cpu = sums.detach().cpu()
        counts_cpu = counts.detach().cpu().to(torch.float32)

        packed = torch.zeros((2, n_cell_lines, d_dict), dtype=torch.float32)
        packed[0] = sums_cpu
        packed[1, :, 0] = counts_cpu  # store counts in first feature column
        return packed

    def unpack_control_sums_counts(
        self: Evaluation, packed: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Inverse of control_sums_counts_by_cell_line packing.
        """
        if packed.ndim != 3 or packed.shape[0] != 2:
            raise ValueError(
                f"Unexpected packed control stats shape: {tuple(packed.shape)}"
            )
        sums = packed[0]
        counts = packed[1, :, 0].to(torch.long)
        return sums, counts

    def control_means_by_cell_line(
        self: Evaluation,
        *,
        config: PerturbationConfig | None = None,
        show_progress: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """
        Returns (means, counts) on CPU:
          means: (n_cell_lines, d_dict) float32
          counts: (n_cell_lines,) long
        """
        cfg = config or self.perturbation_config
        packed = self.cached.control_sums_counts_by_cell_line(
            config=cfg, show_progress=show_progress
        )
        sums, counts = self.unpack_control_sums_counts(packed)
        denom = counts.clamp(min=1).to(torch.float32).unsqueeze(1)
        means = sums / denom
        means = torch.where(
            counts.unsqueeze(1) > 0, means, torch.full_like(means, torch.nan)
        )
        return means, counts

    # ---------------------------------------------------------------------
    # Drug deltas (Act 1 core primitive)
    # ---------------------------------------------------------------------

    @cache_version(1)
    @torch.inference_mode()
    def compute_drug_deltas_tensor(
        self: Evaluation,
        drug: str,
        cell_lines: list[str] | None = None,
        *,
        config: PerturbationConfig | None = None,
        show_progress: bool = True,
    ) -> Tensor:
        """
        Act 1 Step 1.1:
          Returns (n_cell_lines, n_doses, d_dict) where entries are:
            delta[cl, dose_idx, :] = mean_acts(drug @ dose) - mean_acts(control @ control_dosage)

        Notes:
          - "cell_lines" controls *which* cell lines are returned and in which order.
            If None, returns all known cell lines (excluding specials if translator exists).
          - Doses from config (default: (1,2,3)), dose_idx corresponds to that ordering.
        """
        cfg = config or self.perturbation_config
        root = self._meta_root()
        drug_meta = root.metadata_store[cfg.drug_key]
        cell_meta = root.metadata_store[cfg.cell_line_key]
        dose_meta = root.metadata_store[cfg.dosage_key]
        assert (
            isinstance(drug_meta, Tensor)
            and isinstance(cell_meta, Tensor)
            and isinstance(dose_meta, Tensor)
        )

        drug_id = self.get_metadata_id(cfg.drug_key, drug)

        # Determine returned cell_lines
        if cell_lines is None:
            # Prefer translator order; fallback to unique IDs
            try:
                cell_lines = self.get_all_cell_lines(exclude_special=True, config=cfg)
            except Exception:
                ids = cell_meta.unique(sorted=True).tolist()
                cell_lines = [str(i) for i in ids]

        cell_line_ids = torch.tensor(
            [self.get_metadata_id(cfg.cell_line_key, cl) for cl in cell_lines],
            dtype=torch.long,
        )

        # Control means by cell_line (CPU)
        control_means_cpu, _control_counts = self.control_means_by_cell_line(
            config=cfg, show_progress=show_progress
        )

        n_cell_lines_total = control_means_cpu.shape[0]
        d_dict = control_means_cpu.shape[1]
        doses = cfg.doses
        n_doses = len(doses)

        device = self.device

        # Move control means to device for subtraction/indexing
        control_means = control_means_cpu.to(device)

        sums = torch.zeros(
            (n_cell_lines_total * n_doses, d_dict), dtype=torch.float32, device=device
        )
        counts = torch.zeros(
            (n_cell_lines_total * n_doses,), dtype=torch.long, device=device
        )

        # Dose LUT: maps actual dose value -> dose_idx in [0..n_doses-1]
        max_dose = int(max(doses)) if doses else 0
        lut = torch.full((max_dose + 1,), -1, dtype=torch.long, device=device)
        for i, d in enumerate(doses):
            if d < 0:
                raise ValueError(f"Negative dose not supported: {d}")
            if d > max_dose:
                raise RuntimeError("unreachable")
            lut[int(d)] = int(i)

        for doc_ids, agg in self.iter_seq_aggregated_acts_by_chunk(
            pooling=cfg.pooling, device=device, show_progress=show_progress
        ):
            did = doc_ids.to(torch.long)
            d = drug_meta[did].to(device)
            c = cell_meta[did].to(device)
            z = dose_meta[did].to(device)

            m = d == drug_id
            if not m.any():
                continue

            # Select only desired doses (and exclude control dosage)
            z_sel = z[m]
            if z_sel.numel() == 0:
                continue

            # If dosage values exceed LUT range, skip those entries
            in_range = (z_sel >= 0) & (z_sel <= max_dose)
            if not in_range.any():
                continue

            c_sel = c[m][in_range]
            a_sel = agg[m][in_range]
            z_sel = z_sel[in_range]

            # Exclude control dosage explicitly
            if cfg.control_dosage is not None:
                keep = z_sel != int(cfg.control_dosage)
                if not keep.any():
                    continue
                c_sel = c_sel[keep]
                a_sel = a_sel[keep]
                z_sel = z_sel[keep]

            dose_idx = lut[z_sel]
            keep2 = dose_idx >= 0
            if not keep2.any():
                continue

            c_sel = c_sel[keep2]
            a_sel = a_sel[keep2]
            dose_idx = dose_idx[keep2]

            group = c_sel * n_doses + dose_idx
            sums.index_add_(0, group, a_sel)
            counts += torch.bincount(group, minlength=n_cell_lines_total * n_doses)

        means = sums / counts.clamp(min=1).to(torch.float32).unsqueeze(1)
        means = torch.where(
            counts.unsqueeze(1) > 0,
            means,
            torch.full_like(means, torch.nan),
        )
        means = means.view(n_cell_lines_total, n_doses, d_dict)

        # delta = perturbed - control (broadcast over dose)
        deltas = means - control_means.unsqueeze(1)

        # Return just the requested cell_lines order (CPU)
        out = deltas.index_select(0, cell_line_ids.to(device)).detach().cpu()
        return out

    @cache_version(1)
    def compute_drug_deltas_matrix(
        self: Evaluation,
        drug: str,
        cell_lines: list[str] | None = None,
        *,
        config: PerturbationConfig | None = None,
        show_progress: bool = True,
    ) -> Tensor:
        """
        Act 1 Step 1.1 (reduced):
          Returns (n_cell_lines, d_dict)

        dose_mode (from config):
          - "max": take highest dose in `doses` ordering (default dose 3)
          - "slope": fit linear dose-response slope across `doses`
        """
        cfg = config or self.perturbation_config
        deltas_3d = self.cached.compute_drug_deltas_tensor(
            drug=drug,
            cell_lines=cell_lines,
            config=cfg,
            show_progress=show_progress,
        )  # (n_cell_lines, n_doses, d_dict) on CPU

        if cfg.dose_mode == "max":
            return deltas_3d[:, -1, :]

        if cfg.dose_mode == "slope":
            x = torch.tensor(list(cfg.doses), dtype=torch.float32)
            x = x - x.mean()
            denom = (x * x).sum().clamp(min=1e-8)
            y = torch.nan_to_num(deltas_3d.to(torch.float32), nan=0.0)
            # slope per (cell_line, feature): sum_dose(y * x) / sum(x^2)
            slope = (y * x.view(1, -1, 1)).sum(dim=1) / denom
            return slope

        raise ValueError(f"Unknown dose_mode={cfg.dose_mode!r}")

    # ---------------------------------------------------------------------
    # Drug profiles + similarity (Act 1 Step 1.2)
    # ---------------------------------------------------------------------

    @cache_version(1)
    def compute_drug_profile(
        self: Evaluation,
        drug: str,
        cell_lines: list[str] | None = None,
        *,
        config: PerturbationConfig | None = None,
        show_progress: bool = True,
    ) -> Tensor:
        """
        Aggregate perturbation deltas across cell lines -> (d_dict,).

        normalize_across_cell_lines (from config):
          If True: z-score each feature across cell lines before aggregating.
        """
        cfg = config or self.perturbation_config
        matrix = self.cached.compute_drug_deltas_matrix(
            drug=drug,
            cell_lines=cell_lines,
            config=cfg,
            show_progress=show_progress,
        ).to(torch.float32)  # CPU, (n_cell_lines, d_dict)

        x = torch.nan_to_num(matrix, nan=0.0)

        if cfg.normalize_across_cell_lines:
            mu = x.mean(dim=0)
            sd = x.std(dim=0, unbiased=False).clamp(min=1e-8)
            x = (x - mu) / sd

        if cfg.aggregation == "mean":
            prof = x.mean(dim=0)
        elif cfg.aggregation == "median":
            prof = x.median(dim=0).values
        else:
            raise ValueError(f"Unknown aggregation={cfg.aggregation!r}")

        prof = torch.nan_to_num(prof, nan=0.0)
        return prof

    @cache_version(1)
    def compute_drug_similarity_matrix(
        self: Evaluation,
        drugs: list[str] | None = None,
        cell_lines: list[str] | None = None,
        *,
        mode: SimilarityMode = "profile",
        config: PerturbationConfig | None = None,
        show_progress: bool = True,
    ) -> Tensor:
        """
        Returns (n_drugs, n_drugs) cosine similarity.

        mode:
          - "profile": cosine similarity over (d_dict,) profiles
          - "pattern": cosine similarity over flattened (n_cell_lines, d_dict) deltas

        Warning:
          pattern mode can be very large if you pass many drugs and many cell lines.
        """
        cfg = config or self.perturbation_config
        if drugs is None:
            drugs = self.get_all_drugs(
                exclude_special=True, exclude_control=True, config=cfg
            )

        if mode == "profile":
            profiles: list[Tensor] = []
            it = tqdm.tqdm(drugs, disable=not show_progress)
            for d in it:
                p = self.cached.compute_drug_profile(
                    drug=d,
                    cell_lines=cell_lines,
                    config=cfg,
                    show_progress=False,  # avoid nested bars
                )
                profiles.append(p)
            P = torch.stack(profiles).to(torch.float32)  # (n_drugs, d_dict)
            norms = P.norm(dim=1, keepdim=True).clamp(min=1e-8)
            N = P / norms
            return (N @ N.T).to(torch.float32)

        if mode == "pattern":
            mats: list[Tensor] = []
            it = tqdm.tqdm(drugs, disable=not show_progress)
            for d in it:
                m = self.cached.compute_drug_deltas_matrix(
                    drug=d,
                    cell_lines=cell_lines,
                    config=cfg,
                    show_progress=False,
                ).to(torch.float32)
                mats.append(torch.nan_to_num(m, nan=0.0).flatten())
            M = torch.stack(mats)
            norms = M.norm(dim=1, keepdim=True).clamp(min=1e-8)
            N = M / norms
            return (N @ N.T).to(torch.float32)

        raise ValueError(f"Unknown mode={mode!r}")

    def top_similar_drugs(
        self: Evaluation,
        sim_matrix: Tensor,
        drugs: Sequence[str],
        query: str,
        *,
        k: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Given a precomputed similarity matrix, return k most similar drugs to query.
        """
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
        """
        Which features contribute most positively to similarity between two profiles?

        Returns list of:
          (feature_id, p1, p2, contribution=p1*p2)
        """
        p1 = drug1_profile.to(torch.float32)
        p2 = drug2_profile.to(torch.float32)
        contrib = p1 * p2
        top = contrib.topk(min(k, contrib.numel()), largest=True)
        out: list[tuple[int, float, float, float]] = []
        for fid, c in zip(top.indices.tolist(), top.values.tolist(), strict=True):
            out.append(
                (int(fid), float(p1[fid].item()), float(p2[fid].item()), float(c))
            )
        return out

    # ---------------------------------------------------------------------
    # Cytotox helpers (Act 1 Step 1.3)
    # ---------------------------------------------------------------------

    def validate_cytotox_feature(
        self: Evaluation,
        feature_id: int,
        *,
        token_k: int = 500,
        token_mode: str = "active",
        patch_subset_n: int = 500,
    ):
        """
        Thin wrapper around existing APIs:
          - token enrichment
          - average aggregated patching effect (logit-space effect)

        Returns:
          (token_enrich, logit_effects)
        """
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
        cell_lines: list[str] | None,
        *,
        response_feature: int,
        config: PerturbationConfig | None = None,
        show_progress: bool = True,
    ) -> dict[str, float]:
        """
        Sensitivity signal per cell line = delta(response_feature) under drug (default: highest dose).
        Returns mapping {cell_line_name: sensitivity_value}.
        """
        cfg = config or self.perturbation_config
        if cell_lines is None:
            cell_lines = self.get_all_cell_lines(exclude_special=True, config=cfg)

        deltas = self.cached.compute_drug_deltas_matrix(
            drug=drug,
            cell_lines=cell_lines,
            config=cfg,
            show_progress=show_progress,
        )
        sens = deltas[:, int(response_feature)]
        return {cl: float(sens[i].item()) for i, cl in enumerate(cell_lines)}

    def compute_control_features_by_cell_line(
        self: Evaluation,
        cell_lines: list[str] | None,
        *,
        config: PerturbationConfig | None = None,
        show_progress: bool = True,
    ) -> Tensor:
        """
        Returns (n_cell_lines, d_dict) control mean feature activations per cell line.
        """
        cfg = config or self.perturbation_config
        if cell_lines is None:
            cell_lines = self.get_all_cell_lines(exclude_special=True, config=cfg)

        means, _counts = self.control_means_by_cell_line(
            config=cfg, show_progress=show_progress
        )
        # We stored control means indexed by cell_line_id; select in requested order:
        cell_line_ids = torch.tensor(
            [self.get_metadata_id(cfg.cell_line_key, cl) for cl in cell_lines],
            dtype=torch.long,
        )
        return means.index_select(0, cell_line_ids)

    @cache_version(1)
    def compute_feature_sensitivity_correlation(
        self: Evaluation,
        drug: str,
        cell_lines: list[str] | None,
        *,
        response_feature: int,
        config: PerturbationConfig | None = None,
        show_progress: bool = True,
        eps: float = 1e-8,
    ) -> Tensor:
        """
        Act 2:
          Correlate CONTROL features with sensitivity across cell lines.

        Returns:
          correlations: (d_dict,) Pearson r for each feature.
        """
        cfg = config or self.perturbation_config
        if cell_lines is None:
            cell_lines = self.get_all_cell_lines(exclude_special=True, config=cfg)

        # sensitivity vector (n_cell_lines,)
        sens_map = self.compute_sensitivity_by_cell_line(
            drug=drug,
            cell_lines=cell_lines,
            response_feature=response_feature,
            config=cfg,
            show_progress=show_progress,
        )
        sens = torch.tensor([sens_map[cl] for cl in cell_lines], dtype=torch.float32)

        # control features (n_cell_lines, d_dict)
        control_feats = self.compute_control_features_by_cell_line(
            cell_lines=cell_lines,
            config=cfg,
            show_progress=show_progress,
        ).to(torch.float32)

        # Basic NaN handling
        valid = torch.isfinite(sens)
        if not valid.all():
            sens = sens[valid]
            control_feats = control_feats[valid]

        control_feats = torch.nan_to_num(control_feats, nan=0.0)
        sens = torch.nan_to_num(sens, nan=0.0)

        # Center
        yc = sens - sens.mean()
        Xc = control_feats - control_feats.mean(dim=0)

        # Corr = cov / (std_x * std_y)
        num = (Xc * yc.unsqueeze(1)).sum(dim=0)
        den = (Xc.pow(2).sum(dim=0).sqrt() * yc.pow(2).sum().sqrt()).clamp(min=eps)
        r = num / den
        return r
