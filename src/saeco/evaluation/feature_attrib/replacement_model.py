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

    def encode(self, x):
        # x should be of shape [n_layers, n_tokens, d_layer_data]
        cache = SAECache()
        return self.model.nonlinearity(self.model.pre_encoders(x, cache=cache))

    def decode(self, latents):
        # TODO: x should be of shape [n_tokens, d_layer_data]
        cache = SAECache()
        return self.model.decoder(latents, cache=cache)

    def reconstruct_layer(self, inputs: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Uses the CLT to reconstruct a specific layer of the model.

        inputs should be of shape [d_layer_data * (layer_idx + 1)]
        """
        return self.model.reconstruct_layer(inputs, layer_idx)


@define
class ReplacementModel:
    model: LanguageModel
    clt: CrossLayerTranscoder

    attn_prefix: str
    mlp_prefix: str
    logit_site: str
    n_layers: int

    @property
    def input_mlp_sites(self) -> list[str]:
        return [self.mlp_prefix.format(i) + ".input" for i in range(self.n_layers)]

    @property
    def output_mlp_sites(self) -> list[str]:
        return [self.mlp_prefix.format(i) + ".input" for i in range(self.n_layers)]

    @property
    def attn_sites(self) -> list[str]:
        return [self.attn_prefix.format(i) for i in range(self.n_layers)]

    def get_global_replacement_model_prediction(self, prompt: str) -> torch.Tensor:
        """Computes the output logits on a given prompt using a global replacement
        model, running the CLT on off-distribution activations."""

        input_tokens = self.model.tokenizer(prompt, return_tensors="pt").input_ids

        mlp_inputs = []

        with self.model.trace(input_tokens):
            for i in range(self.n_layers):
                mlp_input = getsite(self.model, self.input_mlp_sites[i]).save()

                mlp_inputs.append(mlp_input)

                acts_re = nnsight.apply(
                    lambda x, layer_idx=i: self.clt.reconstruct_layer(
                        torch.cat(x, dim=-1).float()[0], layer_idx
                    ).to(x[0].dtype),
                    mlp_inputs,
                )

                setsite(self.model, self.output_mlp_sites[i], acts_re)

            logits = getsite(self.model, self.logit_site + ".output").save()

        return logits

    def cache_frozen_activations(self, prompt: str) -> dict[str, torch.Tensor]:
        input_tokens = self.model.tokenizer(prompt, return_tensors="pt").input_ids

        frozen_activations = {}

        with self.model.trace(input_tokens):
            for i in range(self.n_layers):
                frozen_activations[self.attn_sites[i]] = getsite(
                    self.model, self.attn_sites[i]
                ).save()
                # TODO: The same for normalization layers...

        return frozen_activations

    def attribute_feature_node(
        self, feature_index: Tuple[int, int, int], prompt: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        frozen_activations = self.cache_frozen_activations(prompt)


def get_active_features(model: LanguageModel, clt: CrossLayerTranscoder, prompt):
    n_layers = 12

    input_prefix = "transformer.h.{}.mlp.input"
    output_prefix = "transformer.h.{}.mlp.output"

    input_sources = [input_prefix.format(i) for i in range(n_layers)]
    output_sources = [output_prefix.format(i) for i in range(n_layers)]

    inputs = []
    outputs = []

    input_tokens = model.tokenizer(prompt, return_tensors="pt").input_ids

    with model.trace(input_tokens):
        for i in range(n_layers):
            input_site = getsite(model, input_sources[i])
            input = input_site.save()
            inputs.append(input)

            output_site = getsite(model, output_sources[i])
            output = output_site.save()
            outputs.append(output)

    inputs = torch.cat(inputs, dim=-1).cuda()[0]
    outputs = torch.cat(outputs, dim=-1).cuda()[0]

    encodings = clt.encode(inputs)
    print(encodings.shape)
    decodings = clt.decode(encodings)[
        -1, :, :
    ]  # Just get the last Matryoshka nesting output

    print(decodings.shape)

    active_indices = torch.nonzero(encodings > 0)
    # active_indices[:, 0] = layer indices
    # active_indices[:, 1] = token indices
    # active_indices[:, 2] = feature indices

    print(active_indices.shape)


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
        attn_prefix="transformer.h.{}.attn",
        mlp_prefix="transformer.h.{}.mlp",
        logit_site="lm_head",
        n_layers=12,
    )

    logits = model.get_global_replacement_model_prediction(
        "The president of the United States is"
    )

    print(logits.shape)


#    get_active_features(model, clt, "The president of the United States is")


if __name__ == "__main__":
    main()
