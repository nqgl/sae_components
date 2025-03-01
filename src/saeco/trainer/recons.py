import einops
import nnsight
import torch

from saeco.data.model_cfg import ActsDataConfig
from saeco.misc.nnsite import getsite, setsite
from saeco.trainer.trainable import Trainable


# @torch.no_grad()
@torch.inference_mode()
def get_recons_loss(
    model,
    encoder: Trainable,
    tokens=None,
    num_batches=10,
    cfg: ActsDataConfig = None,
    bos_processed_with_hook=False,
    batch_size=1,
    cast_fn=...,
):
    cfg = cfg or encoder.cfg
    loss_list = []

    with_sae = to_losses(
        with_sae_runner(model, encoder, cfg, skip_first=not bos_processed_with_hook)
    )
    zero = to_losses(
        zero_ablated_runner(model, cfg, skip_first=not bos_processed_with_hook)
    )
    normal = to_losses(
        normal_runner(model, cfg, skip_first=not bos_processed_with_hook)
    )
    rand_tokens = tokens[torch.randperm(len(tokens))]
    with cast_fn():
        for i in range(num_batches):
            batch_tokens = rand_tokens[i * batch_size : (i + 1) * batch_size].cuda()
            loss = normal(batch_tokens)
            re = with_sae(batch_tokens)
            zeroed = zero(batch_tokens)
            loss_list.append((loss, re, zeroed))
    losses = torch.tensor(loss_list)
    loss, recons_loss, zero_abl_loss = losses.mean(0).tolist()
    print(loss, recons_loss, zero_abl_loss)
    score = (zero_abl_loss - recons_loss) / (zero_abl_loss - loss)
    print(f"{score:.2%}")
    return {
        "recons_score": score,
        "nats_lost": recons_loss - loss,
        "loss": loss,
        "recons_loss": recons_loss,
        "zero_ablation_loss": zero_abl_loss,
    }


def with_sae_runner(
    model: nnsight.LanguageModel, encoder, cfg: ActsDataConfig, skip_first=False
):
    def saerunner(tokens):
        with model.trace(tokens) as tracer:
            lm_acts = getsite(model, cfg.site)
            acts_re = nnsight.apply(lambda x: encoder(x.float()).to(x.dtype), lm_acts)
            if skip_first:
                patch_in = torch.cat([lm_acts[:, :1], acts_re[:, 1:]], dim=1)
            else:
                patch_in = acts_re
            setsite(model, cfg.site, patch_in)
            out = model.output.save()

        return out

    return saerunner


def zero_ablated_runner(
    model: nnsight.LanguageModel, cfg: ActsDataConfig, skip_first=False
):
    def zrunner(tokens):
        with model.trace(tokens) as tracer:
            lm_acts = getsite(model, cfg.site)
            acts_re = nnsight.apply(torch.zeros_like, lm_acts)
            if skip_first:
                patch_in = torch.cat([lm_acts[:, :1], acts_re[:, 1:]], dim=1)
            else:
                patch_in = acts_re
            setsite(model, cfg.site, patch_in)
            out = model.output.save()
        return out

    return zrunner


def normal_runner(model: nnsight.LanguageModel, cfg: ActsDataConfig, skip_first=False):
    def nrunner(tokens):
        return model.trace(tokens, trace=False)

    return nrunner


def to_losses(model_callable):
    def runner(tokens: torch.Tensor):
        out = model_callable(tokens)
        l = einops.rearrange(
            out.logits[:, :-1], "batch seq vocab -> (batch seq) vocab"
        ).cuda()
        t = einops.rearrange(tokens[:, 1:], "batch seq -> (batch seq)").cuda()
        loss = torch.nn.functional.cross_entropy(l, t)

        return loss

    return runner
