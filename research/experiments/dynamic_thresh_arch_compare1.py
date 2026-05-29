from saeco_research.architectures.dynamic_thresh_prolu.model import DynamicThreshSAE


class DynamicThreshSAE_compare1(DynamicThreshSAE):
    def setup(self):
        import wandb

        assert wandb.run is not None
        wandb.run.tags = ("compare1",)
        return super().setup()
