# %%
import wandb
import wandb.apis
import wandb.data_types
import wandb.util
from saeco.misc import lazycall

# from wandb.data_types
# from wandb.wandb_run import Run
from wandb.apis.public import Run, Sweep, Runs

from typing import Any
import os
import pandas as pd
import numpy as np
import polars as pl

# wandb.login(key=os.environ["WANDB_API_KEY"])
api = wandb.Api()


def tuplicate(l):
    if not isinstance(l, list | tuple | set):
        return l
    return tuple(tuplicate(i) if isinstance(i, list | tuple | set) else i for i in l)


left_key_delim = "\u27e8"
right_key_delim = "\u27e9"


class Key:
    def __init__(self, key):
        self.key = tuplicate(key)

    # def __str__(self):
    #     return self.key

    def __eq__(self, other):
        if isinstance(other, Key):
            return self.key == other.key
        elif isinstance(other, type(self.key)) or isinstance(self.key, type(other)):
            return self.key == other
        return self.nicename == other

    def __hash__(self) -> int:
        return hash(self.key)

    @property
    def nicename(self):
        return "/".join(self.key)

    def __repr__(self):
        if isinstance(self.key, tuple):
            return f"{left_key_delim}{self.nicename}{right_key_delim}"
        return f"{left_key_delim}{self.key}{right_key_delim}"


class ValueTarget(Key):
    def __init__(self, name, minimize=True):
        super().__init__(name)
        self.minimize = minimize


class SweepKey(Key):
    def __init__(self, key, values=[]):
        super().__init__(key)
        self.values = set(values)

    def __mul__(self, other):
        if isinstance(other, SweepKey):
            return SweepKeys([self, other])
        elif isinstance(other, SweepKeys):
            return SweepKeys([self, *other.keys])
        else:
            raise TypeError()


def powerset(s1, s2):
    for i in s1:
        assert i not in s2
    return {{i, j} for i in s1 for j in s2}


class SweepKeys:
    def __init__(self, keys: set[SweepKey]):
        self.keys = set(keys)

    @property
    def space(self):
        n = 1
        for k in self.keys:
            n *= len(k.values)
        return n

    def __getitem__(self, i):
        if isinstance(i, int):

            assert 0 <= i < self.space
            n = 1
            return SetKeys(
                {
                    k: list(k.values)[(i := (i // n)) % (n := len(k.values))]
                    for k in self.keys
                }
            )

    def __len__(self):
        return self.space

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


# class SetKey:
#     def __init__(self, key, values):
#         super().__init__()
#         self.key = key
#         self.values = set(values)

#     def __repr__(self):
#         return repr([f"({self.key.key}\u2208{self.values})"])


class SetKeys:
    def __init__(self, d: dict[Key, Any]):
        super().__init__()
        self.d = d

    def __repr__(self):
        return repr(
            " ".join(
                [
                    f"{left_key_delim}{key.nicename}={value}{right_key_delim}"
                    for key, value in self.d.items()
                ]
            )
        )

    def __call__(self, df):
        for key, value in self.d.items():
            df = df[df[key.key] == value]
        return df


sk1 = SweepKey("a", [11, 12, 13])
sk2 = SweepKey("b", [21, 22, 23, 24])
s = sk1 * sk2
s[3]


# %%


class Sweep:
    def __init__(self, sweep_path):
        self.sweep_path = sweep_path
        self.sweep = api.sweep(sweep_path)
        self.full_cfg_key = "full_cfg"
        self.value_targets = [ValueTarget("cache/L2_loss")]
        # self.sweept_fields: dict[list[str], dict[Any, set[Run]]] = {}

    # def __getitem__(self, key):sk1 = SweepKey("a", [11,12,13])

    @property
    @lazycall
    def sweep_cfg(self) -> dict[list[str], dict[Any, set[Run]]]:
        swept_values = {}

        def search_prefix(d, run, prefix=[]):
            for k, v in d.items():
                if k == self.full_cfg_key:
                    assert prefix == []
                    continue
                if isinstance(v, list):
                    v = tuple(v)
                if isinstance(v, dict):
                    search_prefix(v, run, prefix + [k])
                else:
                    sk = tuple(prefix + [k])
                    if sk not in swept_values:
                        swept_values[sk] = {}
                    if v not in swept_values[sk]:
                        swept_values[sk][v] = set()
                    swept_values[sk][v].add(run)

        for run in self.sweep.runs:
            search_prefix(run.config, run)
        old_keys = list(swept_values.keys())
        for key in old_keys:
            if len(swept_values[key]) == 1:
                del swept_values[key]
        return swept_values

    @property
    def runs(self):
        return self.sweep.runs

    @property
    @lazycall
    def run_sweep_values(self) -> dict[Run, dict[list[str], Any]]:
        run_sweep_values = {}
        for k, vd in self.sweep_cfg.items():
            for v, runs in vd.items():
                for run in runs:
                    if run not in run_sweep_values:
                        run_sweep_values[run] = {}
                    run_sweep_values[run][k] = v
        return run_sweep_values

    @property
    @lazycall
    def sweep_keys(self):
        return list(self.sweep_cfg.keys())

    @property
    @lazycall
    def keys(self) -> list[SweepKey]:
        return [SweepKey(k, list(dv.keys())) for k, dv in self.sweep_cfg.items()]

    @property
    @lazycall
    def df(self):
        return pd.DataFrame(
            [
                {**run.summary, "run": run, **self.run_sweep_values[run]}
                # 2
                for i, run in enumerate(self.runs)
                # for run in runs
            ]
        )

    def analyze_key(self, sweep_key):
        df = self.df
        return [
            dict(
                mean=(grp := df.groupby(sweep_key)[vt.name]).mean(),
                med=grp.median(),
                min=grp.min(),
                max=grp.max(),
                std=grp.std(),
            )
            for vt in self.value_targets
        ]

    def analyze_keys(self, sweep_keys):
        df = self.df
        sweep_kvs = {
            k: list(dv.keys())
            for k, dv in self.sweep_cfg.items()
            # for v, run in dv.items()
        }
        l = {(): ()}
        vcs = [()]
        for k in sweep_keys:
            newl = []
            sweep_key_values = list(set(sweep_kvs[k]))
            for existing_values in vcs:
                for skv in sweep_key_values:
                    newl.append(existing_values + (skv,))
            vcs = newl
        results = {}
        for key in vcs:
            df = self.df
            for k, v in zip(sweep_keys, key):
                df = df[df[k] == v]
            results[key] = {vt: df[vt].mean() for vt in self.value_targets}
        return results

    def analyze_key_dominanc(self, sweep_key: SweepKey, target: ValueTarget):
        other_keys = [k for k in self.keys if k != sweep_key]


sw = Sweep("sae sweeps/5uwxiq76")
# # sweep1.mean_analyze(sweep1.sweep_keys[0])
# # sw.keys[2][]
sks = sw.keys[1] * sw.keys[2]
# sks[2](sw.df)["cache/L2_loss"].mean()
# sw.analyze_keys(sw.sweep_keys[1:])
# %%


class SweepAnalysis:
    def __init__(self, sweep: Sweep, skvs: SweepKeys):
        self.sweep = sweep
        self.skvs = skvs
        # self.swdf = {}

        l = []
        for skv in self.skvs:
            df = skv(self.sweep.df)
            l.append(
                {
                    "sweepkey": skv,
                    **{sk: sv for sk, sv in skv.d.items()},
                    **{target: df[target] for target in self.sweep.value_targets},
                }
            )
        self.df = pd.DataFrame(l)
        self.ax2 = list(self.skvs.keys)
        self.cmap = None
        self.analyze()

    def analyze(self):
        self.df["mean"] = self.df["cache/L2_loss"].apply(lambda x: x.mean())
        self.df["min"] = self.df["cache/L2_loss"].apply(lambda x: x.min())
        self.df["max"] = self.df["cache/L2_loss"].apply(lambda x: x.max())
        self.df["med"] = self.df["cache/L2_loss"].apply(lambda x: x.median())

    def graph(self): ...

    def heatmap(self, target=None, style=True):
        if target is None:
            return self.heatmap(self.sweep.value_targets[0])
        # df = df if df is not None else self.df
        piv = self.df.pivot(index=self.ax2[0], columns=self.ax2[1], values=target)
        if not style:
            return piv
        return piv.style.background_gradient(cmap=self.cmap, axis=None)


sa = SweepAnalysis(sw, sks)


# df = sa.df.copy()
# df["mean"] = sa.df["cache/L2_loss"].apply(appfn)
# df


def st(df):
    return (
        s.format(
            {
                "FIELD1": "{:,.0f}",
            }
        )
        .set_properties(
            **{
                "text-align": "center",
                "border-collapse": "collapse",
                "border": "1px solid",
                "width": "200px",
            }
        )
        .hide(axis="index")
    )


# sa.heatmap(df, appfn, "mean").style.background_gradient(axis=None)

sa.heatmap("mean").set_properties(
    **{
        "text-align": "center",
        "border-collapse": "collapse",
        "border": "1px solid",
        "width": "200px",
    }
)

# %%
sa.heatmap("min")
# %%


# with ui.pyplot(figsize=(3, 2)):
#     x = np.linspace(0.0, 5.0)
#     y = np.cos(2 * np.pi * x) * np.exp(-x)
#     plt.plot(sa.heatmap("max"))
#     # plt.plot(x, y, "-")


# sw.mean_analyze(sw.sweep_keys[1])
# # s = api.sweep("sae sweeps/3vsppcm2")# %%
# sweep1 = Sweep("sae sweeps/mfwai3n2")
# # %%

# sk3 = sw.keys[0] * sw.keys[1]
# sk3.keys.add(sw.keys[2])
# sa = SweepAnalysis(sw, sk3)
# # df["mean"] = sa.df["cache/L2_loss"].apply(appfn)

# sa.heatmap(df, appfn, "mean")

# # %%

# ri = iter(sw.sweep.runs)
# r = next(ri)
# # %%
# # rl = list(s.runs)
# # %%
# r
# # rl
# # %%
# [r]
# # %%