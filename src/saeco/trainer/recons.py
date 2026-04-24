import nnsight
import torch

from saeco.data.config.data_cfg import DataConfig
from saeco.data.subject_model_inputs import SubjectBatchInputs, prepare_subject_batch
from saeco.misc.nnsite import getsite, setsite
from saeco.trainer.trainable import Trainable


def get_recons_loss_with_bos(
    llm,
    sae: Trainable,
    tokens=None,
    num_batches=10,
    data_cfg: DataConfig = None,
    batch_size=1,
    cast_fn=...,
):
    return get_recons_loss(
        llm=llm,
        sae=sae,
        tokens=tokens,
        num_batches=num_batches,
        data_cfg=data_cfg,
        bos_processed_with_hook=True,
        batch_size=batch_size,
        cast_fn=cast_fn,
    )


def get_recons_loss_no_bos(
    llm,
    sae: Trainable,
    tokens=None,
    num_batches=10,
    data_cfg: DataConfig = None,
    batch_size=1,
    cast_fn=...,
):
    return get_recons_loss(
        llm=llm,
        sae=sae,
        tokens=tokens,
        num_batches=num_batches,
        data_cfg=data_cfg,
        bos_processed_with_hook=False,
        batch_size=batch_size,
        cast_fn=cast_fn,
    )


@torch.inference_mode()
def get_recons_loss(
    llm,
    sae: Trainable,
    tokens=None,
    num_batches=10,
    data_cfg: DataConfig = None,
    bos_processed_with_hook=False,
    batch_size=1,
    cast_fn=...,
):
    assert data_cfg is not None
    acts_cfg = data_cfg.model_cfg.acts_cfg
    loss_list = []

    with_sae = to_losses(
        with_sae_runner(llm, sae, acts_cfg, skip_first=not bos_processed_with_hook),
        data_cfg,
    )
    zero = to_losses(
        zero_ablated_runner(llm, acts_cfg, skip_first=not bos_processed_with_hook),
        data_cfg,
    )
    neg = to_losses(
        neg_ablated_runner(llm, acts_cfg, skip_first=not bos_processed_with_hook),
        data_cfg,
    )
    normal = to_losses(
        normal_runner(llm, acts_cfg, skip_first=not bos_processed_with_hook),
        data_cfg,
    )
    rand_tokens = tokens[torch.randperm(len(tokens))]
    with cast_fn():
        for i in range(num_batches):
            raw_batch = rand_tokens[i * batch_size : (i + 1) * batch_size]
            raw_batch = raw_batch.cuda() if hasattr(raw_batch, "cuda") else raw_batch
            prepared = prepare_subject_batch(data_cfg, raw_batch)
            re = with_sae(prepared).mean(0, keepdim=True)
            zeroed = zero(prepared).mean(0, keepdim=True)
            loss = normal(prepared).mean(0, keepdim=True)
            neg_loss = neg(prepared).mean(0, keepdim=True)
            loss_list.append((loss, re, zeroed, neg_loss))
    losses = torch.tensor(loss_list)
    loss, recons_loss, zero_abl_loss, neg_loss = losses.mean(0).tolist()

    print(
        f"""
    bos with hook: {bos_processed_with_hook}
    loss: {loss}
    recons_loss: {recons_loss}
    zero_abl_loss: {zero_abl_loss}
    neg_loss: {neg_loss}
          """
    )
    denom = zero_abl_loss - loss
    if denom < 0:
        score = float("nan")
    else:
        score = (zero_abl_loss - recons_loss) / (zero_abl_loss - loss)
    print(f"{score:.2%}")
    return {
        "recons_score": score,
        "nats_lost": recons_loss - loss,
        "loss": loss,
        "recons_loss": recons_loss,
        "zero_ablation_loss": zero_abl_loss,
    }


def with_sae_runner(model: nnsight.LanguageModel, encoder, cfg, skip_first=False):
    def saerunner(prepared: SubjectBatchInputs):
        with model.trace(*prepared.args, **prepared.kwargs):
            for site in cfg.sites:
                lm_acts = getsite(model, site)
                acts_re = encoder(lm_acts).to(lm_acts.dtype)
                if skip_first:
                    patch_in = torch.cat([lm_acts[:, :1], acts_re[:, 1:]], dim=1)
                else:
                    patch_in = acts_re
                setsite(model, site, patch_in)
            out = model.output.logits.save()

        return out

    return saerunner


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


def neg_ablated_runner(model: nnsight.LanguageModel, cfg, skip_first=False):
    def zrunner(prepared: SubjectBatchInputs):
        with model.trace(*prepared.args, **prepared.kwargs):
            for site in cfg.sites:
                lm_acts = getsite(model, site)
                acts_re = -lm_acts + torch.rand_like(lm_acts) * lm_acts.mean()
                if skip_first:
                    patch_in = torch.cat([lm_acts[:, :1], acts_re[:, 1:]], dim=1)
                else:
                    patch_in = acts_re
                setsite(model, site, patch_in)
            out = model.output.logits.save()
        return out

    return zrunner


def normal_runner(model: nnsight.LanguageModel, cfg, skip_first=False):
    def nrunner(prepared: SubjectBatchInputs):
        with model.trace(*prepared.args, **prepared.kwargs):
            out = model.output.logits.save()
        return out

    return nrunner


def to_losses(model_callable, data_cfg: DataConfig):
    """Wrap a model runner so it returns cross-entropy loss averaged over
    real (non-pad) positions.

    Logits at position t predict token at position t+1; we teacher-force and
    apply the batch's attention-style mask at the *target* positions (the
    shifted mask) so padding doesn't contaminate the recons score."""

    skip_n = data_cfg.recons_loss_skip_first_n_targets

    def runner(prepared: SubjectBatchInputs):
        out = model_callable(prepared)
        tokens = prepared.tokens
        assert tokens is not None, "SubjectBatchInputs.tokens is required for CE loss"
        logits = out[:, :-1]
        targets = tokens[:, 1:]
        if prepared.mask is not None:
            target_mask = prepared.mask[:, 1:]
        else:
            target_mask = torch.ones(
                targets.shape, dtype=torch.bool, device=targets.device
            )
        if skip_n > 0:
            target_mask = target_mask.clone()
            target_mask[:, :skip_n] = False
        logits_flat = logits[target_mask].cuda()
        targets_flat = targets[target_mask].cuda()
        loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
        return loss

    return runner
