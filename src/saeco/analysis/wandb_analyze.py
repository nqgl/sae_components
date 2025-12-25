# %%
import asyncio
from functools import cached_property
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
import wandb.apis
import wandb.apis.public as wapublic
import wandb.data_types
import wandb.util

# from wandb.data_types
# from wandb.wandb_run import Run
from wandb.apis.public import Run, Sweep

from saeco.analysis.run_history import RunHistories

r: Run
histories = RunHistories()
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
        self.shortname_n = 0

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
        if isinstance(self.key, str):
            return self.key
        return "-".join([str(e) for e in self.key])

    def shortname(self):
        if isinstance(self.key, str):
            return self.key
        return "-".join([str(e) for e in self.key[-self.shortname_n - 1 :]])

    def __repr__(self):
        if isinstance(self.key, tuple):
            return f"{left_key_delim}{self.nicename}{right_key_delim}"
        return f"{left_key_delim}{self.key}{right_key_delim}"

    def __str__(self):
        return self.nicename


class ValueTarget(Key):
    def __init__(self, name, minimize=True):
        super().__init__(name)
        self.minimize = minimize


class SweepKey(Key):
    def __init__(self, key, values=[]):
        super().__init__(key)
        self.values = dedup(values)

    def __mul__(self, other):
        if isinstance(other, SweepKey):
            return SweepKeys([self, other])
        elif isinstance(other, SweepKeys):
            return SweepKeys([self, *other.keys])
        else:
            raise TypeError()

    # def __len__(self):
    #     return len(self.values)

    # def __iter__(self):
    #     return iter(self.values)


def powerset(s1, s2):
    for i in s1:
        assert i not in s2
    return {{i, j} for i in s1 for j in s2}


def dedup(l):
    return list(dict.fromkeys(l).keys())


class SweepKeys:
    def __init__(self, keys: list[SweepKey]):
        self.keys = dedup(keys)

    @property
    def space(self):
        n = 1
        for k in self.keys:
            n *= len(k.values)
        return n

    def __repr__(self):
        return repr(self.keys)

    def __getitem__(self, i):
        if isinstance(i, int):
            assert 0 <= i < self.space
            n = 1
            return SetKeys(
                {
                    k: sorted(list(k.values))[(i := (i // n)) % (n := len(k.values))]
                    for k in self.keys
                }
            )

    def __len__(self):
        return self.space

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __mul__(self, other):
        if isinstance(other, SweepKey):
            return SweepKeys({*self.keys, other})
        elif isinstance(other, SweepKeys):
            return SweepKeys({*self.keys, *other.keys})
        else:
            raise TypeError()


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

    def filter(self, df):
        for key, value in self.d.items():
            df = df[df[key.key] == value]
        return df

    def __eq__(self, other):
        if isinstance(other, SetKeys):
            return self.d == other.d
        return False

    def __hash__(self):
        return hash(tuple(self.d.items()))


sk1 = SweepKey("a", [11, 12, 13])
sk2 = SweepKey("b", [21, 22, 23, 24])
s = sk1 * sk2
s[3]


# %%


class Sweep:
    def __init__(self, sweep_path):
        self.sweep_path = sweep_path
        self.sweep: wapublic.Sweep = api.sweep(sweep_path)
        self.sweep.load(force=True)

        len(list(self.sweep.runs))
        self.full_cfg_key = "full_cfg"
        self.value_targets = [
            ValueTarget("cache/L2_loss"),
            ValueTarget("eval/L2_loss"),
            ValueTarget("cache/L0"),
            ValueTarget("eval/L0"),
            ValueTarget("cache/L1"),
            ValueTarget("eval/L1"),
            ValueTarget("recons/no_bos/nats_lost"),
            ValueTarget("recons/with_bos/nats_lost"),
        ]
        # self.sweept_fields: dict[list[str], dict[Any, set[Run]]] = {}
        self.prev_avg_min = 0
        self.top_level_ignore_keys = ["pod_info"]
        # df = self.df
        # self.add_target_history()
        # self.add_target_averages()

    # def __getitem__(self, key):sk1 = SweepKey("a", [11,12,13])

    @cached_property
    def sweep_cfg(self) -> dict[list[str], dict[Any, set[Run]]]:
        swept_values = {}

        def search_prefix(d, run, prefix=[]):
            for k, v in d.items():
                if k == self.full_cfg_key:
                    assert prefix == []
                    continue
                if k in self.top_level_ignore_keys and prefix == []:
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

        for run in self.runs:
            search_prefix(run.config, run)
        old_keys = list(swept_values.keys())
        for key in old_keys:
            if len(swept_values[key]) == 1:
                del swept_values[key]
        return swept_values

    @property
    def runs(self) -> list[wapublic.Run]:
        return list(self.sweep.runs)

    @property
    # @lazycall
    def run_sweep_values(self) -> dict[Run, dict[list[str], Any]]:
        run_sweep_values = {}
        for k, vd in self.sweep_cfg.items():
            for v, runs in vd.items():
                for run in runs:
                    if run not in run_sweep_values:
                        run_sweep_values[run] = {}
                    run_sweep_values[run][k] = v
        return run_sweep_values

    @cached_property
    def sweep_keys(self):
        return list(self.sweep_cfg.keys())

    @cached_property
    def keys(self) -> list[SweepKey]:
        return [SweepKey(k, list(dv.keys())) for k, dv in self.sweep_cfg.items()] + [
            SweepKey("__NULLKEY", [1])
        ]

    @cached_property
    def df(self):
        d = pd.DataFrame(
            [
                {
                    **run.summary,
                    "run": run,
                    **self.run_sweep_values[run],
                    "__NULLKEY": 1,
                }
                # 2
                for i, run in enumerate(self.runs)
                # for run in runs
            ]
        )
        d = pd.DataFrame(
            [
                {
                    **run.summary,
                    "run": run,
                    **self.run_sweep_values[run],
                    "__NULLKEY": 1,
                }
                # 2
                for i, run in enumerate(self.runs)
                # for run in runs
            ]
        )
        d = pd.DataFrame(
            [
                {
                    **run.summary,
                    "run": run,
                    **self.run_sweep_values[run],
                    "__NULLKEY": 1,
                }
                # 2
                for i, run in enumerate(self.runs)
                # for run in runs
            ]
        )
        return d

    def add_target_averages(self, min_step=None, force=False):
        # if "history" not in self.df.columns:
        #     self.add_target_history()
        # try:
        #     min_step = int(min_step)
        # except:
        #     return
        min_step = min_step or self.prev_avg_min
        if not force and min_step == self.prev_avg_min:
            return
        for target_key in self.value_targets:
            target_name = target_key.key
            for agg, aggfn in [
                ("mean", np.mean),
                ("med", np.median),
                ("min", np.min),
                ("max", np.max),
                ("std", np.std),
            ]:
                aggkey = f"{target_name}_{agg}"
                self.df[aggkey] = self.df.apply(
                    self._get_target_aggregation_fn(
                        target=target_name,
                        aggregation_fn=aggfn,
                        min_step=min_step,
                    ),
                    axis=1,
                )
        self.prev_avg_min = min_step

    def _get_target_aggregation_fn(self, target, aggregation_fn, min_step):
        def apply_aggregation(row):
            history = row["history"][target]
            try:
                history = history[history["_step"] >= min_step]
                return aggregation_fn(history[target])
            except:
                return np.nan

        return apply_aggregation

    def remove_invalid_runs(self): ...
    def add_target_history(self):
        print()
        # self.add_target_history_async()
        self.sweep.runs
        # self.sweep.load(force=True)
        len({k.id: 0 for k in self.sweep.runs})
        len({k.id: 0 for k in self.runs})
        self.sweep.runs
        histories.get_runs(self.runs, [vt.key for vt in self.value_targets])
        self.df["history"] = self.df.apply(self._get_target_history, axis=1)
        print()

    async def add_target_history_async(self):
        print("start async hist sync")

        # loop = asyncio.get_event_loop()
        runs = [self.df["run"].iloc[i] for i in range(len(self.df))]
        hists = [{} for _ in range(len(self.df))]
        isdone = [False for _ in range(len(self.df))]
        tasks = [
            self._async_get_target_history(*args, done=isdone)
            for args in zip(runs, hists, range(len(self.df)))
        ]
        await asyncio.gather(*tasks)
        print("done async hist sync")

    async def _async_get_target_history(self, run, hist_dict, i, done):
        print("start get")
        history = hist_dict
        for target in self.value_targets:
            key = target.key
            if key in history:
                continue
            run: Run
            # th = run.history(keys=[key, "_step"], samples=2000)
            # hist = th
            # run
            th = run.scan_history(keys=[key, "_step"])
            hist = [i for i in th]

            # hist["t"] = hist["_step"]
            # hist["step"] = hist["_step"]
            # hist["value"]  = hist[key]

            history[key] = hist
        done[i] = True
        print("got run")

    def _get_target_history(self, row):
        run = row["run"]
        # run = row
        self.sweep
        history = {}
        if "history" in self.df.columns:
            history = row["history"]
        for target in self.value_targets:
            key = target.key
            if key in history:
                continue
            run: Run

            # th = run.scan_history(keys=[key, "_step"], min_step=45_000)
            th = histories.load(run, key)

            hist = th

            # hist["t"] = hist["_step"]
            # hist["step"] = hist["_step"]
            # hist["value"]  = hist[key]

            history[key] = hist
        print("got run")
        return history

        print()

        # for targ in self.value_targets:
        #     key = targ.key
        #     if key + "_mean" in self.df.columns:
        #         continue
        #     for run in self.df["run"]:

    def add_target(self, target):
        if isinstance(target, str):
            target = ValueTarget(target)
        self.value_targets.append(target)
        # self.add_target_history(min_step=self.prev_avg_min)
        self.add_target_averages(force=True)

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
            sweep_key_values = dedup(sweep_kvs[k])
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
    def __init__(
        self,
        sweep: Sweep,
        xkeys: SweepKeys,
        ykeys: SweepKeys,
        target: ValueTarget = ValueTarget("cache/L2_loss"),
    ):
        if isinstance(target, str):
            target = ValueTarget(target)
        self.sweep = sweep

        self.xkeys = xkeys
        self.ykeys = ykeys
        # self.swdf = {}

        l = []
        # for skv in self.xkeys * self.ykeys:
        #     df = skv(self.sweep.df)
        #     l.append(
        #         {
        #             "sweepkey": skv,
        #             **{sk: sv for sk, sv in skv.d.items()},
        #             **{target: df[target] for target in self.sweep.value_targets},
        #         }
        #     )
        # self.sweep.add_target_history()
        self.sweep.add_target_averages()

        for xkeyv in self.xkeys:
            dfx = xkeyv.filter(self.sweep.df)
            for ykeyv in self.ykeys:
                df = ykeyv.filter(dfx)
                l.append(
                    {
                        "xkeystate": xkeyv,
                        "ykeystate": ykeyv,
                        "xkeystatestr": str(xkeyv),
                        "ykeystatestr": str(ykeyv),
                        **{xsk: xsv for xsk, xsv in xkeyv.d.items()},
                        **{ysk: ysv for ysk, ysv in ykeyv.d.items()},
                        # **{target: df[target] for target in self.sweep.value_targets},
                        **{target: df[target]},
                    }
                )

        self.df = pd.DataFrame(l)
        # self.ax2 = list(self.skvs.keys)
        self.cmap = None
        self.target = target
        self.analyze()

    def analyze(self):
        self.df["mean"] = self.df[self.target.key].apply(lambda x: x.mean())
        self.df["min"] = self.df[self.target.key].apply(lambda x: x.min())
        self.df["max"] = self.df[self.target.key].apply(lambda x: x.max())
        self.df["med"] = self.df[self.target.key].apply(lambda x: x.median())

    def add_graph_labels(self):
        ccd = {}

        def clash_check(key, value, s):
            k = (key, s)
            if k not in ccd:
                ccd[k] = value
            else:
                if ccd[k] != value:
                    raise ValueError(f"clash {k} {ccd[k]} {value}")

        def uniqi(i):
            return "." + " " * i

        def uniq():
            i = 2
            while True:
                yield uniqi(i)
                i += 1

        keys = list(self.xkeys.keys)
        # keys.reverse()
        prev = dict.fromkeys(keys, "")
        labels = []
        miss_d = {}
        u = uniq()

        def miss(k):
            if k not in miss_d:
                miss_d[k] = next(u)
            return miss_d[k]

        for keystate in self.df["xkeystate"]:
            l = []
            for key in keys:
                v = keystate.d[key]
                if isinstance(v, float):
                    s = (
                        f"{v:.0e}".replace("E-0", "e-")
                        .replace("E+0", "e+")
                        .replace("E", "e")
                    )
                else:
                    s = str(v)
                clash_check(key, v, s)
                # s = f'{keystate.d[key]:.1e.1}'
                if prev[key] == s:
                    l.append(miss((key, keystate.d[key])))
                else:
                    l.append(s)
                prev[key] = s
            labels.append("\n".join(l))

        self.df["label"] = labels

    def plot(self):
        self.add_graph_labels()
        df_exploded = self.df.explode(self.target.key)

        # Create the plot
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Scatter plot for series_col
        prev_color = None
        prev = None
        colors = plt.cm.hsv(np.linspace(0, 1, len(self.df["label"])))
        unused_colors = []
        for label, color in zip(df_exploded["label"].unique(), colors):
            data = df_exploded[df_exploded["label"] == label]
            print()
            cur = tuple(data["xkeystate"].iloc[0].d.items())[1:]
            print(cur)
            if cur == prev and prev_color is not None:
                unused_colors.append(color)
                color = prev_color
            prev = cur
            prev_color = color
            ax1.scatter(
                data["label"],
                data[self.target.key],
                # label=f"Scatter {label}",
                color=color,
                alpha=0.6,
            )

        colors = plt.cm.Accent(np.linspace(0, 1, 4))

        ax1.set_xlabel("Label")
        ax1.set_ylabel(self.target.key, color="b")
        ax1.tick_params(axis="y", labelcolor="b")

        # Line plot for value
        # ax2 = ax1.twinx()
        for agg, color in zip(["mean", "min", "max", "med"], colors):
            ax1.plot(
                self.df["label"],
                self.df[agg],
                color=color,
                marker=".",
                label=agg,
                alpha=0.5,
            )
        # ax2.set_ylabel("Value", color="r")
        # ax2.tick_params(axis="y", labelcolor="r")

        # Title and legend
        plt.title("plot")
        lines1, labels1 = ax1.get_legend_handles_labels()
        # lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1, labels1, loc="upper left")
        # ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        plt.tight_layout()
        plt.show()

    # def graph(self):

    def heatmap(self, target=None, style=True, color_axis=None):
        if target is None:
            return self.heatmap(self.sweep.value_targets[0])
        # df = df if df is not None else self.df
        # piv = self.df.pivot(index=self.xkeys, columns=self.ykeys, values=target)
        piv = self.df.pivot(
            index=[key for key in self.ykeys.keys],
            columns=[key for key in self.xkeys.keys],
            values=target,
        )
        # piv = self.df.pivot(
        #     index=next(iter(self.xkeys.keys)),
        #     columns=next(iter(self.ykeys.keys)),
        #     values=target,
        # )
        if not style:
            return piv
        return piv.style.background_gradient(cmap=self.cmap, axis=color_axis)


# sa = SweepAnalysis(sw, sks)


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

# sa.heatmap("mean").set_properties(
#     **{
#         "text-align": "center",
#         "border-collapse": "collapse",
#         "border": "1px solid",
#         "width": "200px",
#     }
# )

# # %%
# sa.heatmap("min")
# # %%


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
if __name__ == "__main__":
    sweep = Sweep("sae sweeps/5uwxiq76")
    k = sweep.keys[0]
    k1, k2, k3 = sweep.keys

    sa = SweepAnalysis(sweep, SweepKeys([k2, k1]), SweepKeys([]))
    sa.plot()
# %%
