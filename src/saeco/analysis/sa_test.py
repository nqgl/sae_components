# %%
from saeco.analysis.wandb_analyze import (
    Key,
    ValueTarget,
    SweepKey,
    SweepKeys,
    SetKeys,
    SweepAnalysis,
    Sweep,
)
import saeco.analysis.wandb_analyze
import importlib


# %%
# rel()
sweep_keys_list = [
    SweepKey(("key1", 5), [11, 12]),
    SweepKey(("key2", 5), [21, 22, 23]),
    SweepKey(("key3", 5), [31, 32, 33, 34]),
]

sk1, sk2, sk3 = sweep_keys_list

sks12: SweepKeys = sk1 * sk2
sks12


# %%
sks123: SweepKeys = sks12 * sk3


# %%
sks123[5]
# %%
sweep = Sweep("sae sweeps/5uwxiq76")
k = sweep.keys[0]
k1, k2, k3 = sweep.keys
sk1 = k1 * k1
sk2 = k2 * k2
sk3 = k3 * k3


# %%
sa = SweepAnalysis(sweep, sk1, sk2 * sk3)
sa.heatmap("mean")

# %%
