# %%
import wandb
import wandb.apis
import wandb.data_types
import wandb.util
import wandb.wandb_run

api = wandb.Api()

# run = api.run("sae_all/scsae_comparisons/2x7z1z5o")
sweep1 = api.sweep("sae_all/scsae_comparisons/n796532b")
sweep2 = api.sweep("sae_all/scsae_comparisons/qrzgmfzh")
sweep3 = api.sweep("sae_all/scsae_comparisons/30oypo1o")
# %%
runs1 = [run for run in sweep1.runs]
runs2 = [run for run in sweep2.runs]
runs3 = [run for run in sweep3.runs]
len(runs1), len(runs2), len(runs3)
# %%

# def add_nice_name(run):
#     nice_name = run.config["sae_type"]
#     if run.config["sparsity_penalty_type"] == "l1_sqrt":
#         nice_name = "Sqrt(" + nice_name + ")"
#     run.config["nice_name"] = nice_name
#     run.update()


# for run in runs1:
#     add_nice_name(run)


def add_final_nats_lost(run):
    if run.summary.get("recons_final/with_bos/loss") is None:
        raise ValueError("No recons_final/with_bos/loss")
    run.summary["final_nats_lost"] = run.summary["recons_final/with_bos/loss"]
    run.update()


# %%
import numpy as np
import pandas as pd

allruns = [runs1, runs2, runs3]
runs_df = pd.DataFrame(
    [
        {**run.summary, **run.config, "run": run, "experiment": i}
        for i, runs in enumerate(allruns)
        for run in runs
    ]
)
# %%
from_history = {
    "recons_loss": "recons_final/with_bos/recons_loss",
    "baseline_loss": "recons_final/with_bos/loss",
    "recons_score": "recons_final/with_bos/recons_score",
}
from_config = ["sae_type", "sparsity_penalty_type", "lr"]
added_averaged_values = list(from_history.keys()) + ["nats_lost"]
runs_df[added_averaged_values] = None
# hist =


def get_final_history(row):
    run = row["run"]
    assert run.state == "finished"
    history = run.history(keys=list(from_history.values()))
    for k, v in from_history.items():
        history[k] = history.pop(v)
    history["nats_lost"] = history.apply(
        lambda row: row["recons_loss"] - row["baseline_loss"], axis=1
    )
    return history


runs_df["final_history"] = runs_df.apply(
    get_final_history,
    axis=1,
)

for k in added_averaged_values:
    runs_df[k] = runs_df.apply(lambda row: np.mean(row["final_history"][k]), axis=1)


def add_means(row):
    history = row["final_history"]
    mean_finals = np.mean(history, axis=0)
    assert not any([row[k] != None for k in added_averaged_values])
    for k in added_averaged_values:
        row[k] = mean_finals[k]


# %%
import matplotlib.pyplot as plt

runs_df["recons_score"]

# added_averaged_values
# %%
runs_df.loc[28, "final_history"]
runs_df.loc[28, "nats_lost"]

# %%
L0 = dict(
    x=("l0", "L0"),
    xrange=[0, 55],
)
L0y = dict(
    y=("l0", "L0"),
    yrange=[0, 100],
)

L1 = dict(
    x=("l1", "L1"),
    xrange=[15, 60],
)
L1y = dict(
    y=("l1", "L1"),
    yrange=[15, 60],
)
L2 = dict(
    y=("l2_norm", "Error L2 Norm"),
    yrange=[6, 12],
)
nats = dict(
    y=("nats_lost", "CE Loss Degradation (Nats)"),
    yrange=[0, 0.8],
    ytitle="Degradation",
)
nats2 = dict(
    y=("nats_lost", "GPT2 Loss Degradation (Nats)"),
    yrange=[0, 0.8],
)

mse = dict(
    y=("mse", "Mean Squared Error (MSE)"),
    yrange=[0, 0.2],
)
recons_score = dict(
    y=("recons_score", "Reconstruction Score"),
    yrange=[0.94, 1],
)

# %%

spacings = [
    "$\\hspace{36pt}$",
    "$\\hspace{10pt}$",
    "$\\hspace{9.5pt}$",
]

legend_spacings = [
    "$\,\\hspace{0pt}$ ",
    " $\\hspace{26pt}$ ",
]
LEGEND_TITLE = (
    legend_spacings[0]
    + "$\\underline{\\text{Nonlinearity}}$"
    + legend_spacings[1]
    + "$\\underline{\\text{Penalty}}$"
)

RELU = "ReLU}$" + spacings[1]
STE = "STE}\\,$ " + spacings[2]


runs_df["nicer_type"] = runs_df.apply(
    lambda row: (
        f"$\\text{{ProLU}}_{{{RELU if row['sae_type'] == 'SCSAE_RegGrads' else STE}"
        if row["sae_type"] != "VanillaSAE"
        else "$\\text{ReLU}$" + spacings[0]
    ),
    axis=1,
)
runs_df["nicer_penalty"] = runs_df.apply(
    lambda row: "L1" if row["sparsity_penalty_type"] == "l1" else "Sqrt(L1)", axis=1
)
runs_df["nicer_name"] = runs_df.apply(
    lambda row: f"{row['nicer_type']}     {row['nicer_penalty']}", axis=1
)


# plt.scatter(runs_df["l0"], runs_df["nats_lost"])

# %%
import matplotlib as mpl

plt.rc("text", usetex=True)
mpl.rcParams["text.latex.preamble"] = "\\usepackage{amsmath}"

CB_color_cycle = [
    "#4daf4a",
    "#999999",
    "#984ea3",
    "#377eb8",
    "#f781bf",
    "#e41a1c",
    "#a65628",
    "#ff7f00",
    "#dede00",
]
m = {}


def get_color(cat):
    if cat not in m:
        m[cat] = len(m)
    return CB_color_cycle[m[cat]]


def scatter_with_categorical_legend(
    df,
    x,
    y,
    categ="nicer_name",
    xrange=[0, 50],
    yrange=[0, 1],
    title=None,
    af=None,
    pretitle="",
    posttitle="",
    set_legend=False,
    figsize=(10, 5),
    xtitle=None,
    ytitle=None,
):
    if isinstance(x, tuple):
        x, xname = x
    else:
        xname = x
    if isinstance(y, tuple):
        y, yname = y
    else:
        yname = y
    if title is None:
        title = f"{ytitle or yname} vs {xtitle or xname}"
    fig, ax = af or plt.subplots(layout="constrained")

    # for categ in categs:
    for cat in df[categ].unique():
        # print(cat)
        df_cat = df[df[categ] == cat]
        ax.scatter(
            df_cat[x],
            df_cat[y],
            label=cat,
            c=get_color(cat),
            marker="," if all(df_cat["sparsity_penalty_type"] == "l1_sqrt") else "o",
        )
    ax.title.set_text(pretitle + title + posttitle)

    # plt.show()

    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)
    if set_legend or not af:
        legend = fig.legend(
            loc="outside right upper",
            fancybox=True,
            shadow=True,
            title=LEGEND_TITLE,
        )
        fig.set_size_inches(*figsize)
    return fig


scatter_with_categorical_legend(
    runs_df,
    **L0,
    **nats,
)
# %%


def graphs(df):
    scatter_with_categorical_legend(
        df,
        **L0,
        **nats,
    )
    scatter_with_categorical_legend(
        df,
        **L0,
        **recons_score,
    )
    scatter_with_categorical_legend(
        df,
        **L0,
        **L2,
    )
    scatter_with_categorical_legend(
        df,
        **L0,
        **mse,
    )
    scatter_with_categorical_legend(
        df,
        **L1,
        **L2,
    )
    scatter_with_categorical_legend(
        df,
        **L0,
        **L1y,
    )


graphs(runs_df)


# %%
def sep_lr(**kwargs):
    fig, axs = plt.subplots(2, layout="constrained")
    scatter_with_categorical_legend(
        runs_df[runs_df["lr"] == 1e-3],
        **kwargs,
        af=(fig, axs[0]),
        set_legend=True,
        posttitle=f"; $lr = \\text{{{1e-3}}}$",
        figsize=(10, 10),
    )
    scatter_with_categorical_legend(
        runs_df[runs_df["lr"] == 3e-4],
        **kwargs,
        af=(fig, axs[1]),
        posttitle=f"; $lr = \\text{{{3e-4}}}$",
    )


def sep_exp(**kwargs):
    fig, axs = plt.subplots(3, layout="constrained")

    scatter_with_categorical_legend(
        runs_df[runs_df["experiment"] == 0],
        **kwargs,
        pretitle="Sweep 1: ",
        af=(fig, axs[0]),
        set_legend=True,
        figsize=(10, 12),
    )

    scatter_with_categorical_legend(
        runs_df[runs_df["experiment"] == 1],
        **kwargs,
        pretitle="Sweep 2: ",
        af=(fig, axs[1]),
    )

    scatter_with_categorical_legend(
        runs_df[runs_df["experiment"] == 2],
        **kwargs,
        pretitle="Sweep 3: ",
        af=(fig, axs[2]),
    )


def sep_all_exp(**kwargs):
    fig, axs = plt.subplots(4, layout="constrained")

    scatter_with_categorical_legend(
        runs_df,
        **kwargs,
        pretitle="All Sweeps: ",
        af=(fig, axs[0]),
        set_legend=True,
        figsize=(10, 14),
    )

    scatter_with_categorical_legend(
        runs_df[runs_df["experiment"] == 0],
        **kwargs,
        pretitle="Sweep 1: ",
        af=(fig, axs[1]),
    )

    scatter_with_categorical_legend(
        runs_df[runs_df["experiment"] == 1],
        **kwargs,
        pretitle="Sweep 2: ",
        af=(fig, axs[2]),
    )

    scatter_with_categorical_legend(
        runs_df[runs_df["experiment"] == 2],
        **kwargs,
        pretitle="Sweep 3: ",
        af=(fig, axs[3]),
    )
    return fig


def sep_all(**kwargs):
    a = scatter_with_categorical_legend(
        runs_df,
        **kwargs,
        pretitle="All: ",
    )
    sep_lr(**kwargs)
    sep_exp(**kwargs)


# %%

scatter_with_categorical_legend(
    runs_df[runs_df["lr"] == 1e-3],
    **L1,
    **mse,
)

scatter_with_categorical_legend(
    runs_df[runs_df["lr"] == 1e-3],
    **L0,
    **L1y,
)
# %%
sep_lr(
    **L1,
    **mse,
)
# %%
sep_all(
    **L0,
    **L1y,
)

# %%
sep_all_exp(
    **L0,
    **L1y,
)

sep_all_exp(
    **L1,
    **mse,
).savefig("l1_mse.jpg", bbox_inches="tight", dpi=450)


# %%

graphs(runs_df[runs_df["experiment"] == 2])
# %%


fig, axs = plt.subplots(3, layout="constrained")

scatter_with_categorical_legend(
    runs_df[runs_df["experiment"] == 0],
    **L0,
    **nats,
    pretitle="Experiment 1: ",
    af=(fig, axs[0]),
    set_legend=True,
    figsize=(10, 10),
)

scatter_with_categorical_legend(
    runs_df[runs_df["experiment"] == 1],
    **L0,
    **nats,
    pretitle="Experiment 2: ",
    af=(fig, axs[1]),
)

scatter_with_categorical_legend(
    runs_df[runs_df["experiment"] == 2],
    **L0,
    **nats,
    pretitle="Experiment 3: ",
    af=(fig, axs[2]),
)
# %%
# fig, axes = plt.subplots(2, layout="constrained")
# fig.set_size_inches(8, 4)
# fig.legend(loc="outside right upper")
# fig, axes = plt.subplots(2, layout="constrained")
# fig.set_size_inches(8, 4)
# fig.legend(loc="outside right upper")


# %%
sep_all_exp(
    **L0,
    **nats,
).savefig("l0_nats.jpg", bbox_inches="tight", dpi=450)

# %%
scatter_with_categorical_legend(
    runs_df[
        (
            (runs_df["sparsity_penalty_type"] == "l1")
            & (runs_df["nice_name"] != "SCSAE_RegGrads")
        )
        | (runs_df["sae_type"] == "VanillaSAE")
    ],
    **L0,
    **nats,
).savefig("l0_nats_minimal_compare.jpg", bbox_inches="tight", dpi=450)


# %%
sep_all_exp(
    **L1,
    **mse,
).savefig("l1_mse.jpg", bbox_inches="tight", dpi=450)

# %%
sep_all_exp(
    **L0,
    **L1y,
).savefig("l0_l1.jpg", bbox_inches="tight", dpi=450)

# %%
