from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Iterator

import einops
import torch
import tqdm
from torch import Tensor

if TYPE_CHECKING:
    from ..evaluation import Evaluation


@dataclass
class BlockedResult:
    """Result from a blocked coactivation computation."""

    mat: Tensor  # (d_dict, d_dict) accumulated outer product
    f2sum: Tensor  # (d_dict,) sum of squared values per feature

    def normalize_cosine(self) -> Tensor:
        """Normalize as cosine similarity (in-place) and return the matrix."""
        norms = self.f2sum.sqrt()
        self.mat /= norms.unsqueeze(0)
        self.mat /= norms.unsqueeze(1)
        return self.mat

    def verify_diagonal(self, atol: float = 0.001) -> float:
        """Verify diagonal is ~1.0 for valid cosine similarity. Returns product."""
        diag = self.mat.diag()
        prod = diag[~diag.isnan()].prod()
        return prod.item()


@dataclass
class MultiBlockedResult:
    """Result from blocked computation with multiple matrices."""

    results: dict[str, BlockedResult] = field(default_factory=dict)

    def __getitem__(self, key: str) -> BlockedResult:
        return self.results[key]

    def __setitem__(self, key: str, value: BlockedResult):
        self.results[key] = value


def blocked_coactivation(
    chunks: Iterator,
    d_dict: int,
    prepare_features: Callable[[Tensor], Tensor | dict[str, Tensor]],
    blocks_per_dim: int = 1,
    compute_device: torch.device | str = "cuda",
    out_device: torch.device | str | None = None,
    show_progress: bool = True,
    total_chunks: int | None = None,
) -> BlockedResult | MultiBlockedResult:
    """
    Generic blocked coactivation computation over activation chunks.

    This function handles memory-efficient block-wise computation of outer products
    (coactivation matrices) by dividing the d_dict x d_dict output into blocks.

    Args:
        chunks: Iterator of activation chunks with .acts.value attribute
        d_dict: Dictionary size (number of features)
        prepare_features: Function that transforms raw activations (doc, seq, feat)
            into feature matrix (feat, positions). Can return:
            - Single tensor: computes one outer product
            - Dict of tensors: computes multiple outer products with shared blocking
        blocks_per_dim: Number of blocks per dimension (higher = less memory, more compute)
        compute_device: Device for computation
        out_device: Device for output accumulation (defaults to compute_device)
        show_progress: Whether to show progress bar
        total_chunks: Total number of chunks for progress bar

    Returns:
        BlockedResult if prepare_features returns single tensor
        MultiBlockedResult if prepare_features returns dict of tensors
    """
    if out_device is None:
        out_device = compute_device

    block_size = d_dict // blocks_per_dim

    # Initialize accumulators (will be created on first chunk)
    accumulators: dict[str, BlockedResult] | None = None
    single_key = "__single__"

    chunk_iter = tqdm.tqdm(chunks, total=total_chunks) if show_progress else chunks

    for chunk in chunk_iter:
        acts = chunk.acts.value.to(compute_device).to_dense()
        assert acts.ndim == 3, f"Expected 3D acts, got {acts.ndim}D"

        # Get feature matrices
        features = prepare_features(acts)

        # Normalize to dict format
        if isinstance(features, Tensor):
            features = {single_key: features}

        # Initialize accumulators on first chunk
        if accumulators is None:
            accumulators = {
                name: BlockedResult(
                    mat=torch.zeros(d_dict, d_dict, device=out_device),
                    f2sum=torch.zeros(d_dict, device=out_device),
                )
                for name in features
            }

        # Block-wise accumulation
        for i in range(0, d_dict, block_size):
            i_end = min(i + block_size, d_dict)

            for name, feats_mat in features.items():
                block_i = feats_mat[i:i_end]
                # Accumulate squared norms for this block
                accumulators[name].f2sum[i:i_end] += (
                    block_i.pow(2).sum(-1).to(out_device)
                )

                # Compute outer product blocks
                for j in range(blocks_per_dim):
                    j_start = j * block_size
                    j_end = min(j_start + block_size, d_dict)
                    block_j = feats_mat[j_start:j_end]

                    accumulators[name].mat[i:i_end, j_start:j_end] += (
                        block_i @ block_j.T
                    ).to(out_device)

    # Handle empty chunks case
    if accumulators is None:
        raise ValueError("No chunks were processed")

    # Return appropriate type
    if single_key in accumulators and len(accumulators) == 1:
        return accumulators[single_key]
    return MultiBlockedResult(results=accumulators)


class Coactivity:
    @torch.inference_mode()
    def activation_cosims(
        self: "Evaluation",
        out_device: torch.device | str | None = None,
        blocks_per_dim: int = 1,
    ) -> Tensor:
        """
        Compute cosine similarity matrix between features at sequence level.

        Args:
            out_device: Device for output (defaults to self.cuda)
            blocks_per_dim: Number of blocks per dimension for memory efficiency
        """
        if out_device is None:
            out_device = self.cuda

        def prepare(acts: Tensor) -> Tensor:
            return einops.rearrange(acts, "doc seq feat -> feat (doc seq)")

        result = blocked_coactivation(
            chunks=self.saved_acts.chunks,
            d_dict=self.d_dict,
            prepare_features=prepare,
            blocks_per_dim=blocks_per_dim,
            compute_device=self.cuda,
            out_device=out_device,
            total_chunks=len(self.saved_acts.chunks),
        )

        mat = result.normalize_cosine()
        prod = result.verify_diagonal()
        print("prod", prod)
        return mat

    @torch.inference_mode()
    def masked_activation_cosims(
        self: "Evaluation",
        out_device: torch.device | str | None = None,
        blocks_per_dim: int = 1,
        threshold: float = 0,
    ) -> Tensor:
        """
        Compute masked cosine similarities matrix.
        Indexes are like: [masking feature, masked feature]

        Args:
            out_device: Device for output (defaults to self.cuda)
            blocks_per_dim: Number of blocks per dimension for memory efficiency
            threshold: Activation threshold for masking
        """
        if out_device is None:
            out_device = self.cuda

        block_size = self.d_dict // blocks_per_dim
        mat = torch.zeros(self.d_dict, self.d_dict, device=out_device)
        f2sum = torch.zeros(self.d_dict, device=out_device)
        maskedf2sum = torch.zeros(self.d_dict, self.d_dict, device=out_device)

        for chunk in tqdm.tqdm(
            self.saved_acts.chunks, total=len(self.saved_acts.chunks)
        ):
            acts = chunk.acts.value.to(self.cuda).to_dense()
            feats_mat = einops.rearrange(acts, "doc seq feat -> feat (doc seq)")
            feats_sq = feats_mat.pow(2)

            for i in range(0, self.d_dict, block_size):
                i_end = min(i + block_size, self.d_dict)
                block_i = feats_mat[i:i_end]
                block_i_sq = feats_sq[i:i_end]
                block_mask = (block_i > threshold).float()

                f2sum[i:i_end] += block_i_sq.sum(-1).to(out_device)

                for j in range(blocks_per_dim):
                    j_start = j * block_size
                    j_end = min(j_start + block_size, self.d_dict)
                    block_j = feats_mat[j_start:j_end]
                    block_j_sq = feats_sq[j_start:j_end]

                    mat[i:i_end, j_start:j_end] += (block_i @ block_j.T).to(out_device)
                    maskedf2sum[i:i_end, j_start:j_end] += (
                        block_mask @ block_j_sq.T
                    ).to(out_device)

        norms = f2sum.sqrt()
        mat /= maskedf2sum.sqrt()
        mat /= norms.unsqueeze(1)
        prod = mat.diag()[~mat.diag().isnan()].prod()
        assert prod < 1.001 and prod > 0.999
        return mat

    @torch.inference_mode()
    def coactivations(
        self: "Evaluation",
        doc_agg: float | int | str | None = None,
        out_device: torch.device | str | None = None,
        blocks_per_dim: int = 1,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute feature co-occurrence counts and cosine similarity.

        Args:
            doc_agg: Document aggregation mode (None, float/int for Lp norm, "count", "max")
            out_device: Device for output (defaults to self.cuda)
            blocks_per_dim: Number of blocks per dimension for memory efficiency

        Returns:
            (coact_counts, sims): Binary co-occurrence counts and cosine similarities
        """
        if out_device is None:
            out_device = self.cuda

        def prepare(acts: Tensor) -> dict[str, Tensor]:
            feature_activity = self.sequelize(acts, doc_agg=doc_agg)
            feature_bin = (feature_activity > 0).float()
            return {"sims": feature_activity, "counts": feature_bin}

        result = blocked_coactivation(
            chunks=self.saved_acts.chunks,
            d_dict=self.d_dict,
            prepare_features=prepare,
            blocks_per_dim=blocks_per_dim,
            compute_device=self.cuda,
            out_device=out_device,
            total_chunks=len(self.saved_acts.chunks),
        )

        sims_mat = result["sims"].normalize_cosine()
        prod = result["sims"].verify_diagonal()
        assert prod < 1.001 and prod > 0.999

        return result["counts"].mat, sims_mat

    def top_coactivating_features(
        self: "Evaluation", feature_id: int, top_n: int = 10, mode: str = "seq"
    ) -> tuple[Tensor, Tensor]:
        """
        Get top N co-activating features for a given feature.

        Args:
            feature_id: The feature to find co-activations for
            top_n: Number of top features to return
            mode: "seq" for sequence-level or "doc" for document-level

        Returns:
            (indices, values): Top feature indices and their similarity scores
        """
        if mode == "seq":
            mat = self.cached_call.activation_cosims()
        elif mode == "doc":
            mat = self.cached_call.doc_level_co_occurrence()
        else:
            raise ValueError("mode must be 'seq' or 'doc'")

        vals = mat[feature_id].clone()
        vals[feature_id] = -torch.inf
        vals[vals.isnan()] = -torch.inf
        top = vals.topk(top_n + 1)

        v = top.values
        i = top.indices
        v = v[i != feature_id][:top_n]
        i = i[i != feature_id][:top_n]
        return i, v

    @torch.inference_mode()
    def doc_level_co_occurrence(
        self: "Evaluation",
        pooling: str = "mean",
        out_device: torch.device | str | None = None,
        blocks_per_dim: int = 1,
        threshold: float = 0,
    ) -> Tensor:
        """
        Compute document-level co-occurrence with specified pooling.

        Args:
            pooling: "mean", "max", or "binary"
            out_device: Device for output (defaults to self.cuda)
            blocks_per_dim: Number of blocks per dimension for memory efficiency
            threshold: Threshold for binary pooling
        """
        if out_device is None:
            out_device = self.cuda

        def prepare(acts: Tensor) -> Tensor:
            if pooling == "mean":
                acts_pooled = acts.mean(1)
            elif pooling == "max":
                acts_pooled = acts.max(1).values
            elif pooling == "binary":
                acts_pooled = (acts > threshold).sum(1).float()
            else:
                raise ValueError(f"Invalid pooling: {pooling}")
            return einops.rearrange(acts_pooled, "doc feat -> feat doc")

        result = blocked_coactivation(
            chunks=self.saved_acts.chunks,
            d_dict=self.d_dict,
            prepare_features=prepare,
            blocks_per_dim=blocks_per_dim,
            compute_device=self.cuda,
            out_device=out_device,
            total_chunks=len(self.saved_acts.chunks),
        )

        mat = result.normalize_cosine()
        prod = result.verify_diagonal()
        assert prod < 1.001 and prod > 0.999
        return mat

    def sequelize(
        self: "Evaluation",
        acts: Tensor,
        doc_agg: float | int | str | None = None,
    ) -> Tensor:
        """
        Transform 3D activations to 2D feature matrix.

        Args:
            acts: (doc, seq, feat) tensor
            doc_agg: Aggregation mode for documents:
                - None: flatten to (feat, doc*seq)
                - float/int: Lp norm aggregation
                - "count": count of activations per doc
                - "max": max activation per doc

        Returns:
            (feat, positions) tensor
        """
        assert acts.ndim == 3
        if doc_agg:
            if isinstance(doc_agg, float | int):
                acts = acts.pow(doc_agg).sum(dim=1).pow(1 / doc_agg)
            elif doc_agg == "count":
                acts = (acts > 0).sum(dim=1).float()
            elif doc_agg == "max":
                acts = acts.max(dim=1).values
            else:
                raise ValueError("Invalid doc_agg")
            return einops.rearrange(acts, "doc feat -> feat doc")
        else:
            return einops.rearrange(acts, "doc seq feat -> feat (doc seq)")

    def cosims(
        self: "Evaluation",
        doc_agg: float | int | str | None = None,
        out_device: torch.device | str | None = None,
        blocks_per_dim: int = 1,
    ) -> Tensor:
        """Convenience method to get just the cosine similarities from coactivations."""
        return self.coactivations(
            doc_agg=doc_agg, out_device=out_device, blocks_per_dim=blocks_per_dim
        )[1]

    def coactivity(
        self: "Evaluation",
        doc_agg: float | int | str | None = None,
        out_device: torch.device | str | None = None,
        blocks_per_dim: int = 1,
    ) -> Tensor:
        """Convenience method to get coactivation counts and cache cosims."""
        res = self.coactivations(
            doc_agg=doc_agg, out_device=out_device, blocks_per_dim=blocks_per_dim
        )
        self.artifacts[f"cosims({(doc_agg,)}, {{}})"] = res[1]
        return res[0]
