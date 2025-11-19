# %%

from saeco.analysis.wandb_analyze import (
    Sweep,
    SweepAnalysis,
    SweepKey,
    SweepKeys,
)

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
sa = SweepAnalysis(sweep, SweepKeys([k2, k1]), SweepKeys([]))
sa.heatmap("mean")
import seaborn as sns

sa.plot()
# %%%


def uniqi(i):
    return "." + " " * i
    opts = [" ", "\n"]
    s = ""
    n = len(opts)
    while i > 1:
        s += opts[i % n]
        i //= n


def uniq():
    i = 2
    while True:
        yield uniqi(i)
        i += 1


u = uniq()
for i in range(6):
    print(next(u))


def ukey():
    d = {}
    u = uniq()

    def get(k):
        if k not in d:
            d[k] = next(u)
        return d[k]

    return get


# %%
ccd = {}


def clash_check(key, value, s):
    k = (key, s)
    if k not in ccd:
        ccd[k] = value
    else:
        if ccd[k] != value:
            raise ValueError(f"clash {k} {ccd[k]} {value}")


sa = SweepAnalysis(sweep, SweepKeys([k2]), sk3)

keys = list(reversed(sa.xkeys.keys))
keys.reverse()
prev = dict.fromkeys(keys, "")
labels = []
miss = ukey()
for keystate in sa.df["xkeystate"]:
    l = []

    for key in keys:
        v = keystate.d[key]
        if isinstance(v, float):
            s = f"{v:.0e}".replace("E-0", "e-").replace("E+0", "e+").replace("E", "e")
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


fname = "\n".join([str(k) for k in keys])
df = sa.df
df[fname] = labels
sdf = sa.sweep.df
sdf[fname] = labels

# list(ks.d.keys())[0]


# %%
ax = sns.scatterplot(x=sdf[fname], y=sdf["cache/L2_loss"])
# %%
# add mean line
# ax.axhline(df["cache/L2_loss"].mean(), color="red")
ax.figure


ax.plot(df[fname], df["mean"], color="red")
ax.figure
# %%
# add mean grouped by df[fname]
dfgrp = df.groupby(df[fname])
pltlabels = [e[0] for e in dfgrp[fname]]
# ax.plot(df[fname]. , color="red")
# ax.plot(df.groupby(df[fname])["cache/L2_loss"].mean())
# sns.scatterplot(data=sa.df, x="xkeystatestr", y="mean")
# %%
ax.figure
# %%
