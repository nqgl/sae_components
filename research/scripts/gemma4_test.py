# %%
#
from transformers import Gemma4ForConditionalGeneration, Gemma4TextModel

from saeco.data.config.data_config_definitions import gemma_4_lmsys_chat

cfg = gemma_4_lmsys_chat(14, num_train_tokens=1_000_000)

model = cfg.model_cfg.model.cuda()
# %%
toks = cfg.tokens_data(split=cfg.testsplit).get_tokens()

toks.shape
t0 = toks[0:1]
# %%
from saeco.data.subject_model_inputs import SubjectBatchInputs
from saeco.trainer.recons import prepare_subject_batch

NTH = 55
prepared = prepare_subject_batch(cfg, toks[NTH : NTH + 1])
i = NTH + 1
while True:
    p2 = prepare_subject_batch(cfg, toks[i : i + 1])
    if p2.mask[prepared.mask].all():
        break
    i += 1
# %%
import torch

with cfg.model_cfg.autocast_context():
    with torch.inference_mode():
        with model.trace(*prepared.args, **prepared.kwargs):
            inp = model.model.language_model.layers[14].input.save()
model.model.language_model.layers[14].hidden_size_per_layer_input

# %%
layer = model.model.language_model.layers[14]
layer.per_layer_input_gate
# %%


def losses(prepared: SubjectBatchInputs, out, skip_n: int):
    tokens = prepared.tokens
    assert tokens is not None, "SubjectBatchInputs.tokens is required for CE loss"
    logits = out[:, :-1]
    targets = tokens[:, 1:]
    if prepared.mask is not None:
        target_mask = prepared.mask[:, 1:]
    else:
        target_mask = torch.ones(targets.shape, dtype=torch.bool, device=targets.device)
    if skip_n > 0:
        target_mask = target_mask.clone()
        target_mask[:, :skip_n] = False
    logits_flat = logits[target_mask]
    targets_flat = targets[target_mask].cuda()
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    return loss, targets_flat


with cfg.model_cfg.autocast_context():
    with torch.inference_mode():
        with model.trace(*prepared.args, **prepared.kwargs):
            out = model.output.logits.save()
        l = losses(prepared, out, skip_n=256)
        print(l)


# %%
model.tokenizer.decode(prepared.tokens)
# %%
model.tokenizer.decode(l[1])
# %%


with cfg.model_cfg.autocast_context():
    with torch.inference_mode():
        with model.trace(*prepared.args, **prepared.kwargs):
            model.model.language_model.layers[14].input = (
                0 * model.model.language_model.layers[14].input
            )

            out = model.output.logits.save()
        l = losses(prepared, out, skip_n=256)
        print(l)
# %%
import nnsight

from saeco.misc.nnsite import getsite, setsite
from saeco.trainer.recons import to_losses


def normal_runner(model: nnsight.LanguageModel, cfg, skip_first=False):
    def nrunner(prepared: SubjectBatchInputs):
        with model.trace(*prepared.args, **prepared.kwargs):
            out = model.output.logits.save()
        return out

    return nrunner


def zero_ablated_runner(model: nnsight.LanguageModel, cfg, skip_first=False):
    def zrunner(prepared: SubjectBatchInputs):
        with model.trace(*prepared.args, **prepared.kwargs):
            for site in cfg.sites:
                lm_acts = getsite(model, site)
                acts_re = torch.zeros_like(lm_acts)
                if skip_first:
                    patch_in = torch.cat([lm_acts[:, :1], acts_re[:, 1:]], dim=1)
                else:
                    patch_in = acts_re
                setsite(model, site, patch_in)
                out = model.output.logits.save()
        return out

    return zrunner


def justrandom_runner(model: nnsight.LanguageModel, cfg, skip_first=False):
    def zrunner(prepared: SubjectBatchInputs):
        with model.trace(
            p2.args[0],
            *prepared.args[1:],
            **prepared.kwargs,
        ):
            # model.model.language_model.norm.input = torch.randn_like(
            #     model.model.language_model.norm.input
            # )
            out = model.output.logits.save()
        return out

    return zrunner


def zero_ablated_runner2(model: nnsight.LanguageModel, cfg, layer_num: int):
    def zrunner(prepared: SubjectBatchInputs):
        with model.trace(*prepared.args, **prepared.kwargs):
            # for site in cfg.sites:
            # lm_acts = model.model.language_model.layers[15].input
            # acts_re = torch.zeros_like(lm_acts)
            # if skip_first:
            #     patch_in = torch.cat([lm_acts[:, :1], acts_re[:, 1:]], dim=1)
            # else:
            #     patch_in = acts_re
            model.model.language_model.layers[layer_num].input = torch.zeros_like(
                model.model.language_model.layers[layer_num].input
            )
            # setsite(model, site, patch_in)
            out = model.output.logits.save()
        return out

    return zrunner


nr = to_losses(normal_runner(model, cfg.model_cfg.acts_cfg, skip_first=False), cfg)

zr = to_losses(
    zero_ablated_runner(model, cfg.model_cfg.acts_cfg, skip_first=False), cfg
)

zr2 = to_losses(zero_ablated_runner2(model, cfg.model_cfg.acts_cfg, layer_num=12), cfg)
jrr = to_losses(justrandom_runner(model, cfg.model_cfg.acts_cfg, skip_first=False), cfg)

# %%

with cfg.model_cfg.autocast_context():
    with torch.inference_mode():
        print("nr", nr(prepared).mean())
        print("zr", zr(prepared).mean())
        print("zr2", zr2(prepared).mean())
        print("jrr", jrr(prepared).mean())
# %%
zr_i_l = [
    to_losses(zero_ablated_runner2(model, cfg.model_cfg.acts_cfg, layer_num=i), cfg)
    for i in range(len(model.model.language_model.layers))
]
# %%


with cfg.model_cfg.autocast_context():
    with torch.inference_mode():
        for i, l in enumerate(zr_i_l):
            print(i, l(prepared).mean())


# %%
tcfg = model.config.text_config
list(enumerate(tcfg.layer_types))
# %%
tcfg.num_kv_shared_layers

# %%
"""
> was trying to figure out mysterious behavior where zero ablated loss was ~= to normal loss
        > not a thing I've seen before
figured it out I think:
(re e2b)
here is my current understanding
layers 0-14 are original attention layers
    (13 is a sliding window attn)
    (14 is global)
after that, it reuses the kv from 13/14
so, if you ablate <=13, this fully breaks the connection to input tokens
ablate 14 and you half-break it
>=15 and there is still a path to the inputs

yeah, that makes good sense of the curve I observe here:
"""

with cfg.model_cfg.autocast_context():
    with torch.inference_mode():
        for i, l in enumerate(zr_i_l):
            print(i, l(prepared).mean())
"""
I think the mystery is resolved.
"""
