import wandb


def initialize_sweep(d):
    sweep_id = wandb.sweep(
        sweep=d,
        project="sweep_test",
    )
    f = open("sweep/sweep_id.txt", "w")
    f.write(sweep_id)
    f.close()


class CategSweep:
    

def run():
    wandb.init()
    scfg = ConfigFromSweep(**wandb.config)
    cfg, lgcfg = get_configs_from_sweep(scfg=scfg)
    # if cfg.sae_cfg.sae_type != "VanillaSAE":
    #     cfg.neuron_dead_threshold = -1
    cfg.l1_coeff = cfg.l1_coeff * sparsity_coeff_adjustment(scfg)
    wandb.config.update({"adjusted_l1_coeff": cfg.l1_coeff})
    if QUICK_TEST:
        cfg.buffer_cfg.batch_size = 512
        print("\nwarning: quick test")
        print("small batches")
    if wandb.config["l0_target"]:
        trainer = L0TargetingTrainer(
            cfg,
            model=test.model,
            val_tokens=test.val_tokens,
            legacy_cfg=lgcfg,
            target_l0=wandb.config["l0_target"],
            adjust_eps=0.0003,
        )
    else:
        trainer = Trainer(
            cfg,
            model=test.model,
            val_tokens=test.val_tokens,
            legacy_cfg=lgcfg,
        )

    wandb.config.update(trainer.cfg)
    nice_name = wandb.config["sae_type"]
    if wandb.config["sparsity_penalty_type"] == "l1_sqrt":
        nice_name = "Sqrt(" + nice_name + ")"
    wandb.config.update({"nice_name": nice_name})
    trainer.train(
        cfg.data_cfg.train_data_batch_generator(
            model=test.model,
            batch_size=cfg.buffer_cfg.batch_size,
            nsteps=None if not QUICK_TEST else 20_000,
        )
    )
    wandb.finish()


sweep_id = open("sweep/sweep_id.txt").read().strip()

SWEEP_NAME = "scsae_comparisons"


def main():
    wandb.agent(
        sweep_id,
        function=run,
        entity="sae_all",
        project=SWEEP_NAME,
    )


if __name__ == "__main__":
    main()
