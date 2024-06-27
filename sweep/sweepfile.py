# %%
from saeco.architectures.gate_hierarch import (
    hierarchical_softaux,
    HierarchicalSoftAuxConfig,
    HGatesConfig,
)
from saeco.trainer.runner import TrainingRunner, TrainConfig, RunConfig
from saeco.sweeps import Swept, Sweeper

# PROJECT = "nn.Linear Check"
# train_cfg = TrainConfig(
#     l0_target=45,
#     coeffs={
#         "sparsity_loss": 3e-4,
#         "L2_loss": 10,
#     },
#     lr=Swept[float](1e-3, 3e-4),
#     use_autocast=True,
#     wandb_cfg=dict(project=PROJECT),
#     l0_target_adjustment_size=0.001,
#     batch_size=4096,
#     use_lars=True,
#     betas=Swept[tuple[float, float]](
#         (0.9, 0.99),
#         (0.9, 0.999),
#     ),
# )
# acfg = HierarchicalSoftAuxConfig(
#     num_levels=Swept[int](1, 2),
#     aux0=Swept[bool](True, False),
#     hgates=HGatesConfig(
#         l1_scale_base=1,
#         num_levels=2,
#         BF=Swept[int](2**5, 2**4),
#         untied=True,
#         classic=Swept[bool](True, False),
#         penalize_inside_gate=False,
#         target_hierarchical_l0_ratio=Swept[float](0.5, 0.25),
#         relu_gate_encoders=False,
#     ),
# )

PROJECT = "nn.Linear Check"
train_cfg = TrainConfig(
    l0_target=45,
    coeffs={
        "sparsity_loss": 3e-4,
        "L2_loss": 1,
    },
    lr=1e-3,
    use_autocast=True,
    wandb_cfg=dict(project=PROJECT),
    l0_target_adjustment_size=0.001,
    batch_size=4096,
    use_lars=True,
    betas=(0.9, 0.99),
)
acfg = HierarchicalSoftAuxConfig(
    num_levels=2,
    aux0=Swept[bool](True, False),
    hgates=HGatesConfig(
        l1_scale_base=1,
        num_levels=2,
        BF=2**4,
        untied=True,
        classic=Swept[bool](True, False),
        penalize_inside_gate=Swept[bool](True, False),
        target_hierarchical_l0_ratio=0.5,
        relu_gate_encoders=False,
    ),
)
cfg = RunConfig[HierarchicalSoftAuxConfig](
    train_cfg=train_cfg,
    arch_cfg=acfg,
)


def run(cfg):
    tr = TrainingRunner(cfg, hierarchical_softaux)
    tr.trainer.train()


def main():
    # sweeper = Sweeper()
    import argparse
    import sys

    swfpath = sys.argv[0]
    print(swfpath)
    thispath = "/".join(swfpath.split("/")[:-1])
    filename = swfpath.split("/")[-1]
    thispath = thispath.split("/sae_components/")[-1]
    sw = Sweeper(thispath)
    # sw.start_agent()
    sw.initialize_sweep()
    n = input("create instances?")
    try:
        n = int(n)
    except ValueError:
        n = False
    from ezpod import Pods, RunProject, RunFolder

    pods = Pods.All()
    if n:
        pods.make_new_pods(n)

    pods.sync()
    pods.setup()
    print("running!")
    pods.runpy(f"src/saeco/sweeps/sweeper.py {thispath}", purge_after=True)


if __name__ == "__main__":
    main()
