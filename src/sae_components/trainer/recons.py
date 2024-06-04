import torch
import tqdm
import einops
from sae_components.data.sc.dataset import ActsDataConfig


# @torch.no_grad()
@torch.inference_mode()
def get_recons_loss(
    model,
    encoder,
    buffer,
    all_tokens=None,
    num_batches=5,
    local_encoder=None,
    cfg: ActsDataConfig = None,
    bos_processed_with_hook=False,
    batch_size=64,
):
    cfg = cfg or encoder.cfg
    if local_encoder is None:
        local_encoder = encoder
    loss_list = []
    with torch.autocast(device_type="cuda"):
        for i in range(num_batches):
            tokens = (all_tokens if all_tokens is not None else buffer.all_tokens)[
                torch.randperm(
                    len(all_tokens if all_tokens is not None else buffer.all_tokens)
                )[:batch_size]
            ]
            # assert torch.all(50256 == tokens[:, 0])
            loss = model(tokens, return_type="loss")
            recons_loss = model.run_with_hooks(
                tokens,
                return_type="loss",
                fwd_hooks=[
                    (
                        cfg.hook_site,
                        lambda *a, **k: replacement_hook(
                            *a,
                            **k,
                            encoder=local_encoder,
                            cfg=cfg,
                            bos_processed_with_hook=bos_processed_with_hook,
                        ),
                    )
                ],
            )
            # mean_abl_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(cfg.act_name, mean_ablate_hook)])
            zero_abl_loss = model.run_with_hooks(
                tokens,
                return_type="loss",
                fwd_hooks=[(cfg.hook_site, zero_ablate_hook)],
            )
            loss_list.append((loss, recons_loss, zero_abl_loss))
    losses = torch.tensor(loss_list)
    loss, recons_loss, zero_abl_loss = losses.mean(0).tolist()

    print(loss, recons_loss, zero_abl_loss)
    score = (zero_abl_loss - recons_loss) / (zero_abl_loss - loss)
    print(f"{score:.2%}")
    # print(f"{((zero_abl_loss - mean_abl_loss)/(zero_abl_loss - loss)).item():.2%}")
    return {
        "recons_score": score,
        "loss": loss,
        "recons_loss": recons_loss,
        "zero_ablation_loss": zero_abl_loss,
    }


def replacement_hook(
    acts, hook, encoder, cfg: ActsDataConfig, bos_processed_with_hook=False
):
    acts_shape = acts.shape
    acts_re = acts.reshape(-1, cfg.d_data)
    mlp_post_reconstr = encoder(acts_re.reshape(-1, cfg.d_data))

    mlp_post_reconstr = mlp_post_reconstr.reshape(acts_shape)
    seq_len = acts_shape[1]
    assert seq_len == 128
    # print(acts[:, 0])
    # assert False, acts[0, 0]
    if bos_processed_with_hook:
        return mlp_post_reconstr
    return torch.cat((acts[:, :1], mlp_post_reconstr[:, 1:]), dim=1)


def mean_ablate_hook(mlp_post, hook):
    mlp_post[:, :] = mlp_post.mean([0, 1])
    return mlp_post


def zero_ablate_hook(mlp_post, hook):
    mlp_post[:, :] = 0.0
    return mlp_post
