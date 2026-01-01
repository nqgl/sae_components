from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import einops
import torch
import tqdm
from torch import Tensor

if TYPE_CHECKING:
    from saeco.evaluation.evaluation import Evaluation


def f2sum_fn_default(acts: Tensor) -> Tensor:
    # acts: doc seq feat
    return acts.sum(-2).pow(2).sum(0)


class Coactivity:
    @torch.inference_mode()
    def coacts(
        self: Evaluation,
        S: Callable[[Tensor], tuple[Tensor, Tensor]],
        reduce_prod: Callable[[Tensor, Tensor], Tensor],
        f2sum_fn: Callable[[Tensor], Tensor] = f2sum_fn_default,
        out_device: torch.device | str | None = None,
        f_chunk_i: int | None = None,
        f_chunk_j: int | None = None,
    ) -> Tensor:
        out_device = out_device or self.device
        f_chunk_i = f_chunk_i or self.d_dict
        f_chunk_j = f_chunk_j or self.d_dict

        mat = torch.zeros(self.d_dict, self.d_dict, device=out_device)

        for chunk in tqdm.tqdm(self.cached_acts.chunks, total=len(self.cached_acts.chunks)):
            acts = chunk.acts.value.to(self.device).to_dense()
            if acts.ndim != 3:
                raise ValueError("Expected acts shaped (doc, seq, feat)")

            s0, s1 = S(acts)

            for i in range(0, self.d_dict, f_chunk_i):
                act0s = s0[..., i : i + f_chunk_i]
                _ = f2sum_fn(acts[..., i : i + f_chunk_i])  # retained for potential future use

                for j in range(0, self.d_dict, f_chunk_j):
                    act1s = s1[..., j : j + f_chunk_j]
                    res = reduce_prod(act0s, act1s)
                    mat[i : i + f_chunk_i, j : j + f_chunk_j] += res.to(out_device)

        return mat

    @torch.inference_mode()
    def coacts2(
        self: Evaluation,
        S0: Callable[[Tensor], Tensor],
        S1: Callable[[Tensor], Tensor],
        reduce_prod: Callable[[Tensor, Tensor], Tensor],
        f2sum_fn: Callable[[Tensor], Tensor] = f2sum_fn_default,
        out_device: torch.device | str | None = None,
        f_chunk_i: int | None = None,
        f_chunk_j: int | None = None,
    ) -> Tensor:
        out_device = out_device or self.device
        f_chunk_i = f_chunk_i or self.d_dict
        f_chunk_j = f_chunk_j or self.d_dict

        mat = torch.zeros(self.d_dict, self.d_dict, device=out_device)

        for chunk in tqdm.tqdm(self.cached_acts.chunks, total=len(self.cached_acts.chunks)):
            acts = chunk.acts.value.to(self.device).to_dense()
            if acts.ndim != 3:
                raise ValueError("Expected acts shaped (doc, seq, feat)")

            for i in range(0, self.d_dict, f_chunk_i):
                act0s = S0(acts[..., i : i + f_chunk_i])
                _ = f2sum_fn(acts[..., i : i + f_chunk_i])

                for j in range(0, self.d_dict, f_chunk_j):
                    act1s = S1(acts[..., j : j + f_chunk_j])
                    res = reduce_prod(act0s, act1s)
                    mat[i : i + f_chunk_i, j : j + f_chunk_j] += res.to(out_device)

        return mat

    def causal_coacts(
        self: Evaluation,
        acts_pre_mod_func: Callable[[Tensor], Tensor] = lambda acts: acts,
        out_device: torch.device | str | None = None,
        f_chunk_i: int | None = None,
        f_chunk_j: int | None = None,
    ) -> Tensor:
        @torch.compile(dynamic=True)
        def S1(acts: Tensor) -> Tensor:
            postfix = acts_pre_mod_func(acts).flip(-2).cumsum(-2).flip(-2)
            return postfix

        def reduce_prod(a: Tensor, postfix: Tensor) -> Tensor:
            return einops.einsum(a, postfix, "doc seq f1, doc seq f2 -> f1 f2")

        return self.coacts2(
            S0=acts_pre_mod_func,
            S1=S1,
            reduce_prod=reduce_prod,
            out_device=out_device,
            f_chunk_i=f_chunk_i,
            f_chunk_j=f_chunk_j,
        )

    def act_coacts2(
        self: Evaluation,
        out_device: torch.device | str | None = None,
        f_chunk_i: int | None = None,
        f_chunk_j: int | None = None,
    ) -> Tensor:
        def agg(acts: Tensor) -> Tensor:
            return acts.sum(-2)

        def S(acts: Tensor) -> tuple[Tensor, Tensor]:
            a = agg(acts)
            return a, a

        def reduce_prod(a: Tensor, b: Tensor) -> Tensor:
            return einops.einsum(a, b, "doc f1, doc f2 -> f1 f2")

        return self.coacts(
            S=S,
            reduce_prod=reduce_prod,
            out_device=out_device,
            f_chunk_i=f_chunk_i,
            f_chunk_j=f_chunk_j,
        )