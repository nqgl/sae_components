# %%

from functools import cached_property

import einops
import torch
from attrs import define
from nnsight import LanguageModel
from saeco.architectures.matryoshka_clt import MatryoshkaCLT
from saeco.components.features.features_param import get_featuresparams
from saeco.components.sae_cache import SAECache
from saeco.misc.nnsite import getsite, setsite
from saeco.trainer.trainable import Trainable
from tqdm import tqdm

llm = LanguageModel("openai-community/gpt2", device_map="auto")

# %%


@define
class ReplacementModel:
    model: LanguageModel
    clt: Trainable

    attn_format: str
    mlp_format: str
    embedding_site: str
    ln_formats: list[str] | str

    n_layers: int
    d_data: int
    d_dict: int

    @cached_property
    def attn_sites(self) -> list[str]:
        return [self.attn_format.format(layer) for layer in range(self.n_layers)]

    @cached_property
    def mlp_sites(self) -> list[str]:
        return [self.mlp_format.format(layer) for layer in range(self.n_layers)]

    @cached_property
    def ln_sites(self) -> list[str]:
        if isinstance(self.ln_formats, str):
            self.ln_formats = [self.ln_formats]

        return sum(
            [
                [ln_format.format(layer) for layer in range(self.n_layers)]
                for ln_format in self.ln_formats
            ],
            [],
        )

    def attribute_prompt(self, prompt: str | torch.Tensor):
        if isinstance(prompt, str):
            prompt = self.model.tokenizer(prompt, return_tensors="pt").input_ids

        n_toks = prompt.size(1)

        embed, acts, error_vecs, logits = self.setup_attribution(prompt)

        logit_idx, logit_p = self.compute_important_logits(logits, 10, 0.95)
        n_logits = len(logit_idx)

        # acts.shape is [n_layers, n_tokens, n_feats_per_layer]

        active_feats = torch.nonzero(acts > 0)
        # active_feats[:, 0] is layer index
        # active_feats[:, 1] is token index
        # active_feats[:, 2] is feature index

        # temporarily, cut down the size of the feats
        # TODO: Remove this when my CLT has low L0
        indices = torch.sort(torch.randperm(active_feats.size(0))[:1_000])[0]
        active_feats = active_feats[indices]
        n_active_feats = len(active_feats)

        activation_levels = acts[
            active_feats[:, 0], active_feats[:, 1], active_feats[:, 2]
        ]

        # enc_dirs is a tensor of shape [n_active_feats, d_data]
        enc_dirs = self.get_encoder_dirs(active_feats)

        # dec_dirs is a list of n_active_feats tensors of shape [n_layers - layer, d_data]
        # where layer for index i is active_feats[i, 0]
        dec_dirs = self.get_decoder_dirs(active_feats)

        # attrib_matrix[t, s] represents the attribution of source feature s to target feature t
        n_total_attribs = n_active_feats + self.n_layers * n_toks + n_logits + n_toks
        attrib_matrix = torch.zeros(n_total_attribs, n_total_attribs)

        for i in tqdm(range(len(active_feats)), desc="Attributing features"):
            attrib = self.attribute_feat(
                prompt,
                active_feats,
                activation_levels,
                embed,
                error_vecs,
                i,
                enc_dirs,
                dec_dirs,
                n_logits,
            )
            attrib_matrix[i] = attrib

        # attrib = self.attribute_feat(
        #     prompt,
        #     active_feats,
        #     activation_levels,
        #     embed,
        #     error_vecs,
        #     500,
        #     enc_dirs,
        #     dec_dirs,
        #     n_logits,
        # )

        print(attrib_matrix.shape)
        print(attrib.shape)

    def attribute_feat(
        self,
        prompt,
        active_feats: torch.Tensor,
        activation_levels: torch.Tensor,
        embed: torch.Tensor,
        error_vecs: torch.Tensor,
        t_idx: int,
        enc_dirs: torch.Tensor,
        dec_dirs: torch.Tensor,
        n_logits: int,
    ):
        n_toks = prompt.size(1)
        n_active_feats = len(active_feats)

        t_layer = active_feats[t_idx][0]
        t_token = active_feats[t_idx][1]
        t_feature = active_feats[t_idx][2]

        inject_dir = torch.zeros(1, n_toks, self.d_data).cuda()
        inject_dir[0, t_token, :] = enc_dirs[t_idx]

        resid_grads = []
        with self.model.trace(prompt):
            # TODO: Implement linearizing the model backwards pass of the model
            # (probably will pair with Glen on this)

            # for site in self.attn_sites:
            #     output = getsite(self.model, site).output.save()
            #     output.grad[..., :-768] = 0

            # for site in self.mlp_sites:
            #     output = getsite(self.model, site).output.save()
            #     output.grad = torch.zeros_like(output.grad)

            # for site in self.ln_sites:
            #     output = getsite(self.model, site).output.save()
            #     output.grad = torch.zeros_like(output.grad)

            # Save the residual stream Jacobians
            for site in self.mlp_sites[:t_layer]:
                output = getsite(self.model, site).output.save()
                resid_grads.append(output.grad.save())

            embed_output = getsite(self.model, self.embedding_site).output.save()
            embed_grad = embed_output.grad.save()

            # Backwards with respect to our target encoder direction.
            t_layer_resid = getsite(self.model, self.mlp_sites[t_layer])
            t_layer_resid.input.backward(gradient=inject_dir)

        if t_layer > 0:
            jacob = torch.stack(resid_grads, dim=0)[
                :, 0
            ]  # Remove prompt batch dimension

        feat_attribs = torch.zeros(n_active_feats).cuda()

        embed_jacob = embed_grad[0]  # Remove prompt batch dimension

        # Attribute source features
        # TODO: There is a batched way to do this, but i need to batch per source layer,
        # because the number of decoder features varies by layer.
        # In the current representation of decoder dirs this is not-obvious
        for s_idx in range(n_active_feats):
            s_layer = active_feats[s_idx][0]
            s_token = active_feats[s_idx][1]

            if active_feats[s_idx, 0] >= t_layer:
                continue

            decs = dec_dirs[s_idx][: t_layer - s_layer]
            jacob_slice = jacob[s_layer:, s_token]

            sum_of_dec_dots = torch.einsum("ld,ld->", decs, jacob_slice)

            feat_attribs[s_idx] = activation_levels[s_idx] * sum_of_dec_dots

        # Attribute embeddings
        embed_attrib = torch.einsum("td,td->t", embed, embed_jacob)

        # Attribute error nodes
        if t_layer > 0:
            error_vecs_slice = error_vecs[:t_layer]
            partial_error_attrib = torch.einsum("ltd,ltd->lt", error_vecs_slice, jacob)
            partial_error_attrib = einops.rearrange(
                partial_error_attrib, "layers tokens -> (layers tokens)"
            )

            error_attrib = torch.zeros(self.n_layers * n_toks).cuda()
            error_attrib[: len(partial_error_attrib)] = partial_error_attrib
        else:
            error_attrib = torch.zeros(self.n_layers * n_toks).cuda()

        # Logits cannot affect feature activations
        logit_attribs = torch.zeros(n_logits).cuda()

        return torch.cat([feat_attribs, embed_attrib, error_attrib, logit_attribs])

    def get_encoder_dirs(self, active_feats: torch.Tensor) -> torch.Tensor:
        encs = [fp for fp in get_featuresparams(self.clt) if fp.type == "enc"]
        encs.sort(key=lambda x: int(x.param_id))

        weights = torch.stack([enc.param.data for enc in encs], dim=0)

        encoder_dirs = weights[active_feats[:, 0], active_feats[:, 2], :]
        return encoder_dirs

    def get_decoder_dirs(self, active_feats: torch.Tensor) -> torch.Tensor:
        decs = [fp for fp in get_featuresparams(self.clt) if fp.type == "dec"]
        decs.sort(key=lambda x: int(x.param_id))
        weights = [dec.param.data for dec in decs]

        dirs = []
        for i in tqdm(range(len(active_feats)), desc="Collecting decoder dirs"):
            layer_idx = active_feats[i, 0]
            feat_idx = active_feats[i, 2]

            dirs_for_this_feat = torch.zeros(self.n_layers - layer_idx, self.d_data)
            for decoder_layer in range(layer_idx, self.n_layers):

                # Position in the concatenated features for this decoder
                pos_in_concat = layer_idx * self.d_dict + feat_idx
                dirs_for_this_feat[decoder_layer - layer_idx] = weights[decoder_layer][
                    pos_in_concat, :
                ]

            dirs.append(dirs_for_this_feat.cuda())

        return dirs

    def compute_important_logits(
        self, logits: torch.Tensor, max_n_logits: int, desired_logit_prob: float
    ) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        top_p, top_idx = torch.topk(probs, max_n_logits)

        cutoff = int(torch.searchsorted(torch.cumsum(top_p, 0), desired_logit_prob))
        top_p, top_idx = top_p[:cutoff], top_idx[:cutoff]

        return top_idx, top_p

    def setup_attribution(self, prompt: torch.tensor):
        mlp_inputs = []
        mlp_outputs = []

        with self.model.trace(prompt):
            embed = getsite(self.model, self.embedding_site).output.save()

            for site in self.mlp_sites:
                mlp_inputs.append(getsite(self.model, site).input.save())

                mlp_outputs.append(getsite(self.model, site).output.save())

            logits = self.model.output.logits.save()

        embed = embed[0]
        logits = logits[0, -1, :]

        mlp_inputs = torch.cat(mlp_inputs, dim=-1).cuda()[0]

        cache = SAECache()
        cache.acts = ...
        recons = self.clt(mlp_inputs, cache=cache)[-1]
        recons = torch.stack(torch.chunk(recons, self.n_layers, dim=-1))
        acts = cache.acts

        mlp_outputs = torch.stack(mlp_outputs).squeeze().cuda()

        error_vecs = mlp_outputs - recons

        return (embed, acts, error_vecs, logits)


arch = MatryoshkaCLT.load("../../../../matryoshka_clt_test_model.pt", load_weights=True)
arch.setup()

clt = arch.trainable

replacement_model = ReplacementModel(
    llm,
    clt,
    "transformer.h.{0}.attn.c_attn",
    "transformer.h.{0}.mlp",
    "transformer.wte",
    ["transformer.h.{0}.ln_1", "transformer.h.{0}.ln_2"],
    12,
    768,
    768 * 8,
)

print(replacement_model.attn_sites)
print(replacement_model.mlp_sites)
print(replacement_model.ln_sites)
# print(replacement_model.get_encoder_dirs(None))

replacement_model.attribute_prompt("The Eiffel Tower is in the city of ")

enc_dir = torch.randn(768)

# replacement_model.attribute_dir(enc_dir, 5)
