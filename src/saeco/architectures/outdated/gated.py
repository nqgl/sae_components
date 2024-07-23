import torch
import torch.nn as nn

from saeco.architectures.initialization.initializer import Initializer
from saeco.components.ops.detach import Thresh
import saeco.core as cl
from saeco.core.collections.parallel import Parallel
from saeco.components import (
    L1Penalty,
    EMAFreqTracker,
    L2Loss,
    SparsityPenaltyLoss,
    SAECache,
)

# from saeco.core.linear import Bias, NegBias, Affine, MatMul
from saeco.components.ops.fnlambda import Lambda
from saeco.core.reused_forward import ReuseForward, ReuseCache
from saeco.core import Seq
import saeco.components.features.features as ft

import saeco.components as co
from saeco.trainer.trainable import Trainable


def gated_sae(
    init: Initializer,
    detach=True,
):

    init._encoder.bias = False
    init._encoder.add_wrapper(ReuseForward)
    init._decoder.add_wrapper(ft.NormFeatures)
    init._decoder.add_wrapper(ft.OrthogonalizeFeatureGrads)
    enc_mag = Seq(
        pre_bias=ReuseForward(init._decoder.sub_bias()),
        r_mag=cl.ops.MulParallel(
            identity=ReuseForward(init.encoder),
            exp_r=Lambda(
                func=lambda x: torch.exp(x),
                module=init.dict_bias(),
            ),
        ),
        bias=init.new_encoder_bias().resampled(),
        nonlinearity=nn.ReLU(),
    )

    enc_gate = ReuseForward(
        Seq(
            pre_bias=(
                Parallel(left=cl.ops.Identity(), right=init.decoder.bias).reduce(
                    (lambda l, r: l - r.detach())
                )
                if detach
                else ReuseForward(init._decoder.sub_bias())
            ),
            weight=ReuseForward(init.encoder),
            bias=init.new_encoder_bias().resampled(),
            nonlinearity=nn.ReLU(),
        )
    )

    # models
    gated_model = Seq(
        encoder=cl.Parallel(
            magnitude=enc_mag,
            gate=Thresh(enc_gate),
        ).reduce(
            lambda x, y: x * y,
        ),
        freqs=EMAFreqTracker(),
        metrics=co.metrics.ActMetrics(),
        decoder=init.decoder,
    )

    model_aux = Seq(  # this one is just used for training the gate appropriately
        encoder=enc_gate,  # oh and it's missing 1-2 detaches
        L1=L1Penalty(),
        freqs=EMAFreqTracker(),
        decoder=init._decoder.detached if detach else init.decoder,
    )

    # losses
    losses = dict(
        L2_loss=L2Loss(gated_model),
        L2_aux_loss=L2Loss(model_aux),
        sparsity_loss=SparsityPenaltyLoss(model_aux),
    )
    return [gated_model, model_aux], losses


def gated_sae_no_detach(init):
    return gated_sae(init, detach=False)


def main():
    d_data = 1000 * 5
    d_dict = 500 * 5
    model, losses = gated_sae(d_data, d_dict)
    model = model.cuda()
    print(model)
    print(losses)

    print(model.state_dict())

    x = torch.randn(7000, d_data).cuda()
    cache = SAECache()
    cache += ReuseCache()
    y = model(x, cache=cache)

    print(y)


def test_train(model, losses, data):
    d_data = 768
    d_dict = 8 * d_data
    features = torch.randn(d_dict, d_data).cuda()
    from saeco.trainer.trainer import Trainer
    import tqdm
    import wandb

    trainer = Trainer({}, Trainable([model], losses).cuda())
    batch_size = 4096

    trainer.train(data)


def main():
    d_data = 768
    d_dict = 8 * d_data
    features = torch.randn(d_dict, d_data).cuda()
    model, losses = gated_sae(d_data, d_dict)
    from saeco.trainer.trainer import Trainer
    import tqdm
    import wandb

    trainer = Trainer({}, Trainable([model], losses).cuda())
    batch_size = 4096

    @torch.no_grad()
    def data_generator():
        rand = torch.rand(batch_size, d_dict, device="cuda")
        for i in tqdm.trange(10000):
            rand[:] = rand + 0.001
            x = rand @ features
            yield x

    # for i in data_generator():
    #     pass

    trainer.train(data_generator())


if __name__ == "__main__":
    main()
