from saeco.architectures.gate_hierarch import (
    hierarchical_softaux,
    HierarchicalSoftAuxConfig,
)
from saeco.trainer.runner import TrainingRunner, TrainConfig, RunConfig
from saeco.sweeps import Swept

PROJECT = "nn.Linear Check"
train_cfg = TrainConfig(
    l0_target=45,
    coeffs={
        "sparsity_loss": 3e-4,
        "L2_loss": 10,
    },
    lr=1e-3,
    use_autocast=True,
    wandb_cfg=dict(project=PROJECT),
    l0_target_adjustment_size=0.001,
    batch_size=2048,
    use_lars=True,
    betas=(0.9, 0.99),
)
cfg = RunConfig(
    train_cfg=train_cfg,
    arch_cfg=HierarchicalSoftAuxConfig(),
)


tr = TrainingRunner(cfg, hierarchical_softaux)
tr.trainer.train()
