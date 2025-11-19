# %%
import wandb
from wandb.apis.public import Run, Sweep

api = wandb.Api()

sweep: Sweep = api.sweep("L0Targeting/7jx8xxac")

# %%
r = sweep.runs[0]
# %%
for r in sweep.runs:
    r: Run
    tr = r.config["arch_cfg"]["thresh_range"]
    r.config["arch_cfg"]["thresh_range_upper"] = tr[0]
    r.config["arch_cfg"]["thresh_range_lower"] = tr[1]
    r.summary["thresh_range_upper"] = tr[0]
    r.summary["thresh_range_lower"] = tr[1]
    r.summary["arch_cfg/thresh_range_upper"] = tr[0]
    r.summary["arch_cfg/thresh_range_lower"] = tr[1]

    r.update()
# %%
