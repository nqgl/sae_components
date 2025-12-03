# %%

from typing import Callable
import neptune
import pandas as pd

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)
# %%

project = "nqgl/default-project"
project = neptune.init_project(project, mode="read-only")
orig_runs_table = project.fetch_runs_table().to_pandas()

# %%
CONSTRAINTS = {"cache/L0_": lambda x: x < 53}
TARGET = "cache/L2_loss_"
TARGET = "eval/L2_loss_"
AUX_TARGETS = ["cache/L2_loss_", "eval/L0_"]
MAXIMIZE = False


def apply_constraints(
    runs_table: pd.DataFrame, constraints: dict[str, Callable[[float], bool]]
) -> pd.DataFrame:
    for key, fn in constraints.items():
        runs_table = runs_table.loc[runs_table[key].apply(fn)]
    return runs_table


def get_all_runs_table(orig_runs_table: pd.DataFrame = orig_runs_table) -> pd.DataFrame:
    runs_table = apply_constraints(orig_runs_table, CONSTRAINTS)
    runs_table = runs_table.sort_values(by=TARGET, ascending=not MAXIMIZE)
    experiment_mask = runs_table[
        "config/full_cfg/train_cfg/data_cfg/model_cfg/model_load_cfg/chk_ident/checkpoint/batch"
    ].isna()
    runs_table = runs_table[experiment_mask]
    return runs_table


all_runs_table = get_all_runs_table()
# %%
cfg_keys = [
    "train_cfg/lr",
    "train_cfg/optim",
    "train_cfg/betas/0",
    "train_cfg/betas/1",
    "train_cfg/use_lars",
    "train_cfg/l0_target_adjustment_size",
    "train_cfg/weight_decay",
    "train_cfg/coeffs/sparsity_loss",
    "arch_cfg/thresh_cfg/decay_toward_mean",
    "arch_cfg/thresh_cfg/momentum",
    "arch_cfg/thresh_cfg/l0_diff_mult",
    "arch_cfg/thresh_cfg/lr",
    "arch_cfg/thresh_cfg/initial_value",
    "arch_cfg/thresh_cfg/stepclamp",
    "arch_cfg/thresh_cfg/log_diff",
    "init_cfg/d_data",
    "init_cfg/dict_mult",
    "train_cfg/batch_size",
    "arch_cfg/l1_end_scale",
]


def get_all_keys(runs_table: pd.DataFrame) -> list[str]:
    keys = []
    for ck in cfg_keys:
        found = [k for k in runs_table.keys() if k.lower().endswith(ck.lower())]
        if len(found) == 0:
            print(f"key {ck} not found")
            found = [k for k in runs_table.keys() if ck.lower() in k.lower()]
            print(f"found similar \n\t{'\n\t'.join(found)}")

        elif len(found) > 1:
            print(f"key {ck} found multiple times: {found}")
        keys.extend(found)

    return [TARGET] + AUX_TARGETS + list(CONSTRAINTS.keys()) + keys


all_keys = get_all_keys(all_runs_table)
# %%
# %%
orig_runs_table["config/full_cfg/train_cfg/batch_size"].value_counts()
orig_runs_table["config/full_cfg/init_cfg/d_data"].value_counts()
orig_runs_table["config/full_cfg/init_cfg/dict_mult"].value_counts()

# %%
opts = [
    "config/full_cfg/train_cfg/batch_size",
    "config/full_cfg/init_cfg/d_data",
    "config/full_cfg/init_cfg/dict_mult",
]

# %%


def split_runs_table(runs_table: pd.DataFrame, opts: list[str], K: int = 2):
    topk_run_types = runs_table[opts].value_counts()[:K].index
    ort = runs_table
    run_tables = []
    for r in topk_run_types:
        result = ort[(ort[topk_run_types.names] == r).all(axis=1)]
        print(f"run type {r} has {result.shape[0]} runs")
        run_tables.append(result)
    return run_tables


# %%

# topk_run_types = orig_runs_table[opts].value_counts()[:2]
# rk = topk_run_types.index
# # %%
# rk.names
# rk.values
# # %%
# ort = orig_runs_table
# run_tables = []
# for r in rk:
#     result = ort[(ort[rk.names] == r).all(axis=1)]
#     print(f"run type {r} has {result.shape[0]} runs")
#     run_tables.append(result)
# # %%
run_tables = split_runs_table(all_runs_table, opts, 2)

# %%
run_tables[0][all_keys].head()

# %%
run_tables[1][all_keys].head(15)
# %%
