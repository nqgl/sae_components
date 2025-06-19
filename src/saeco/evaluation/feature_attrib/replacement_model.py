from typing import Tuple

import nnsight

import torch
from attrs import define
from nnsight import LanguageModel

from saeco.architectures.matryoshka_clt import MatryoshkaCLT
from saeco.components.sae_cache import SAECache
from saeco.data.model_cfg import ActsDataConfig, ModelConfig
from saeco.misc.nnsite import getsite, setsite


# Non-trainable CLT used for encoding, decoding, etc
@define
class CrossLayerTranscoder:
    model: MatryoshkaCLT

    def encode(self, x) -> torch.Tensor:
        # x should be of shape [n_layers, n_tokens, d_layer_data]
        cache = SAECache()
        return self.model.nonlinearity(self.model.pre_encoders(x, cache=cache))

    def decode(self, latents) -> torch.Tensor:
        # TODO: x should be of shape [n_tokens, d_layer_data]
        cache = SAECache()
        return self.model.decoder(latents, cache=cache)


@define
class ReplacementModel:
    model: LanguageModel
    clt: CrossLayerTranscoder

    attn_site: str
    mlp_prefix: str

    logit_site: str
    embedding_site: str
    n_layers: int

    d_model: int
    d_dict: int

    @property
    def input_mlp_sites(self) -> list[str]:
        return [self.mlp_prefix.format(i) + ".input" for i in range(self.n_layers)]

    @property
    def output_mlp_sites(self) -> list[str]:
        return [self.mlp_prefix.format(i) + ".input" for i in range(self.n_layers)]

    @property
    def attn_sites(self) -> list[str]:
        return [self.attn_site.format(i) for i in range(self.n_layers)]

    def setup_attribution(
        self, prompt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = []
        outputs = []

        with self.model.trace(prompt):
            embed = getsite(self.model, self.embedding_site).output.save()

            for i in range(self.n_layers):
                input_site = getsite(self.model, self.input_mlp_sites[i])
                input = input_site.save()
                inputs.append(input)

                output_site = getsite(self.model, self.output_mlp_sites[i])
                output = output_site.save()
                outputs.append(output)

            logits = self.model.output.logits.save()

        # Remove batch dimension, just take last logit.
        logits = logits[0, -1, :]

        # Remove the batch dimension
        embed = embed[0]

        real_mlp_acts = torch.stack(outputs).cuda().squeeze()

        mlp_inputs = torch.cat(inputs, dim=-1).cuda()[0]
        clt_encodings = self.clt.encode(mlp_inputs)
        mlp_act_recons = self.clt.decode(clt_encodings)
        mlp_act_recons = torch.stack(
            torch.chunk(mlp_act_recons[-1, :, :], self.n_layers, dim=-1)
        )  # Just get the last Matryoshka nesting output, reshape catted last dim into layers

        error_vecs = real_mlp_acts - mlp_act_recons

        return (embed, clt_encodings, error_vecs, logits)

    def generate_attribution_graph(
        self, prompt: str, max_n_logits: int = 10, desired_logit_prob: float = 0.95
    ):

        prompt = self.model.tokenizer(prompt, return_tensors="pt").input_ids
        n_toks = prompt.size(-1)

        embed, encodings, error_vecs, logits = self.setup_attribution(prompt)

        logit_idx, logit_p = self.compute_salient_logits(
            max_n_logits, desired_logit_prob, logits
        )

        active_features = torch.nonzero(encodings > 0)

        n_active_features = active_features.size(0)

        # active_features[:, 0] = layer indices
        # active_features[:, 1] = token indices
        # active_features[:, 2] = feature indices

        encoder_vectors = torch.stack(
            [
                self.clt.model.encoders[layer_idx].weight[feature_idx]
                for layer_idx, _, feature_idx in active_features
            ]
        )

        # TODO: Figure out decoder access pattern and then decide how to store them.

        feature_idx = 100_000
        selected_feature = active_features[feature_idx]
        encoder_dir = encoder_vectors[feature_idx]
        print(selected_feature)
        print(encoder_dir)

        layer = selected_feature[0]
        token_pos = selected_feature[1]

        with self.model.trace(prompt) as tracer:
            last_layer = getsite(self.model, self.input_mlp_sites[layer])

            for attn_site in self.attn_sites:
                attn_val = getsite(self.model, attn_site).save()
                attn_val.grad.save().detach()

            last_layer_val = last_layer.save()
            last_layer_val.grad[:, token_pos] = encoder_dir

            last_layer.backward(gradient=torch.zeros_like(last_layer))

        print(target_tensor.shape)

        # The attribution matrix is a square matrix.
        # The nodes are ordered as: CLT features, embeds, errors, logits.
        # The number of CLT nodes is the number of active features on the prompt.
        # There are n_toks embed nodes, n_layers * n_toks error nodes, and len(logit_idx) logit nodes.

        # The attribution from target node to source_node node is given by: "graph[target][source]".

    def compute_salient_logits(self, max_n_logits, desired_logit_prob, logits):
        probs = torch.softmax(logits, dim=-1)
        top_p, top_idx = torch.topk(probs, max_n_logits)

        cutoff = int(torch.searchsorted(torch.cumsum(top_p, 0), desired_logit_prob)) + 1
        top_p, top_idx = top_p[:cutoff], top_idx[:cutoff]

        return top_idx, top_p


def main():
    model_cfg = ModelConfig(
        acts_cfg=ActsDataConfig(
            excl_first=True,
            sites=[],
            d_data=768,
            autocast_dtype_str="bfloat16",
            force_cast_dtype_str="bfloat16",
            storage_dtype_str="bfloat16",
        ),
        model_name="gpt2",
    )

    model = model_cfg.model

    raw_clt = MatryoshkaCLT.load(
        "../../../../matryoshka_clt_test_model.pt", load_weights=True
    )
    raw_clt.setup()

    clt = CrossLayerTranscoder(raw_clt)

    model = ReplacementModel(
        model,
        clt,
        attn_site="transformer.h.{}.attn",
        mlp_prefix="transformer.h.{}.mlp",
        logit_site="lm_head",
        embedding_site="transformer.wte",
        n_layers=12,
        d_model=768,
        d_dict=768 * 8,
    )

    model.generate_attribution_graph("The president of the United States is ")


#    get_active_features(model, clt, "The president of the United States is")


if __name__ == "__main__":
    main()
