# saeco

## Installation

Depends on ezpod.

```bash
pip install -e .
```


## Disclaimer 

This project is in the midst of a few transitions.

The system for defining Architectures has been recently overhauled.

The logging system used to depend on Weights and Biases, but is in the middle of transitioning away. This transition is currently in progress so some functionality is broken and the current state of the code around logging is well below my standards. 

This codebase has been mainly used by me personally, so currently most of the code is not documented. I'm beginning to change that, but at the moment I wouldn't advise building on this codebase unless you can ask me questions.

## What this does

This is a library for training sparse autoencoders. You use components from the library, and then you get a bunch of the functionality taken care of for you.
- training pipeline
    - data generation and caching
- resampling "for free"
- L0Targeting
- remote orchestration for running large sweeps
- logging
- sweeping as a first class feature of configs
- other nice stuff



## Examples


### Vanilla SAE

Check out [the vanilla SAE Architecture](src/saeco/architectures/vanilla/vanilla_model.py) for an example of how the library is used to define a model architecture.

```python

class VanillaConfig(SweepableConfig):
    # SweepableConfig is a subclass of pydantic BaseModel
    pre_bias: bool = False
    # this is implicitly bool | Swept | SweepExpression due to being a SweepableConfig


class VanillaSAE(Architecture[VanillaConfig]):
    # setup is called before models are constructed
    def setup(self):
        # these will add wrappers to the decoder that ensure:
        # 1. the features are normalized after each optimizer step to have unit norm
        # 2. the gradients of the features are orthogonalized after each backward pass before the optimizer step
        self.init._decoder.add_wrapper(ft.NormFeatures)
        self.init._decoder.add_wrapper(ft.OrthogonalizeFeatureGrads)

    # model_prop tells the Architecture class that this method
    # is the method that constructs the model.
    # model_prop is a subclass of cached_property, so self.model will always
    # refer to the same instance of the model
    @model_prop
    def model(self):
        return SAE(
            encoder_pre=Seq(
                **useif(self.cfg.pre_bias, pre_bias=self.init._decoder.sub_bias()),
                lin=self.init.encoder,
            ),
            nonlinearity=nn.ReLU(),
            decoder=self.init.decoder,
        )

    # loss_prop designates a Loss that will be used in training
    @loss_prop
    def L2_loss(self):
        return L2Loss(self.model)

    @loss_prop
    def sparsity_loss(self):
        return SparsityPenaltyLoss(self.model)
```

Then see how simple it is [run the grid search on the VanillaSAE](experiments/vanilla_example_training.py).

### Gated SAE

For an example of a more complex Architecture, here's a quick implementation of a GatedSAE:



```python

class GatedConfig(SweepableConfig):
    pre_bias: bool = False
    detach: bool = True


class Gated(Architecture[GatedConfig]):
    def setup(self):
        self.init._encoder.bias = False
        self.init._encoder.add_wrapper(ReuseForward)
        self.init._decoder.add_wrapper(ft.NormFeatures)
        self.init._decoder.add_wrapper(ft.OrthogonalizeFeatureGrads)

    @cached_property
    def enc_mag(self):
        return Seq(
            **useif(
                self.cfg.pre_bias,
                pre_bias=ReuseForward(self.init._decoder.sub_bias()),
            ),
            r_mag=cl.ops.MulParallel(
                identity=ReuseForward(self.init.encoder),
                exp_r=co.Lambda(
                    func=lambda x: torch.exp(x),
                    module=self.init.dict_bias(),
                ),
            ),
            bias=self.init.new_encoder_bias(),
            nonlinearity=nn.ReLU(),
        )

    @cached_property
    def enc_gate(self):
        return Seq(
            **useif(
                self.cfg.pre_bias,
                pre_bias=(
                    cl.Parallel(
                        left=cl.ops.Identity(), right=self.init.decoder.bias
                    ).reduce((lambda l, r: l - r.detach()))
                    if self.cfg.detach
                    else ReuseForward(self.init._decoder.sub_bias())
                ),
            ),
            weight=ReuseForward(self.init.encoder),
            bias=self.init.new_encoder_bias(),
            nonlinearity=nn.ReLU(),
        )

    @model_prop
    def gated_model(self):
        return SAE(
            encoder=cl.Parallel(
                magnitude=self.enc_mag,
                gate=co.ops.Thresh(self.enc_gate),
            ).reduce(
                lambda x, y: x * y,
            ),
            decoder=self.init.decoder,
            penalty=None,
        )

    L2_loss = gated_model.add_loss(L2Loss)

    @aux_model_prop
    def model_aux(self):
        return SAE(
            encoder=self.enc_gate,
            freqs=EMAFreqTracker(),
            decoder=(
                self.init._decoder.detached if self.cfg.detach else self.init.decoder
            ),
        )

    L2_aux_loss = model_aux.add_loss(L2Loss)
    sparsity_loss = model_aux.add_loss(SparsityPenaltyLoss)
```



## Future

This is still a work in progress, there's more to come (both for the library and this README).

Feel free to contact me if you have any questions!