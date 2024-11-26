import asyncio

from nicegui import ui
from saeco.analysis.ddmenuprop import ddmenuprop, ddupdate
from saeco.analysis.uiitem import UIE
from saeco.analysis.wandb_analyze import (
    Sweep,
    SweepAnalysis,
    SweepKey,
    SweepKeys,
    ValueTarget,
)


class SAView:
    def __init__(self, sweep="sae sweeps/5uwxiq76"):
        self.sw = Sweep(sweep)
        self.sa = None
        with ui.card():
            ui.label(sweep)
            try:
                ui.label(
                    self.sw.runs[0].metadata["args"][0].split("/")[-1]
                    + "->"
                    + self.sw.runs[0].metadata["args"][-1]
                )
            except:
                pass
            with ui.row():
                with ui.card() as c:
                    self.heatmap

                with ui.row():
                    self.aggregation
                    with ui.card():
                        ui.label("Selected Keys")
                        self.key1
                        self.key2
                    with ui.card():
                        ui.label("Target")
                        # self.aggregation
                        self.base_target
                        self.new_value_target
                    with ui.card():
                        ui.label("Run Agg")
                        self.target_aggregation
                        self.begin_aggregation_phase

                    with ui.card():
                        self.update_hist_button
                        self.cmap
                        self.color_axis

            self.filters_keys_el = KeyFilters(self.sw.keys)
        self.hist_update()
        self.update()

    @UIE
    def cmap(self, cb):
        i = ui.input(label="cmap", on_change=cb, value="viridis_r")
        # i.value = "viridis_r"
        return i

    @cmap.value
    def cmap(self, e):
        return e.value

    def hist_update(self):
        self.sw.add_target_history()
        if self.begin_aggregation_phase is not ...:
            self.sw.add_target_averages(
                min_step=self.begin_aggregation_phase, force=True
            )

    @UIE
    def update_hist_button(self, cb):
        return ui.button("Update", on_click=self.hist_update)

    def update(self):
        # sa = SweepAnalysis(self.sw, self.key1, self.key2, self.base_target)
        # sa.cmap = self.cmap
        # ddupdate()
        pass

    @UIE
    def new_value_target(self, cb):
        # with ui.card():
        inp = ui.input(label="New Value Target")

        def add_target():
            vt = ValueTarget(inp.value)
            if vt not in self.sw.value_targets:
                self.sw.value_targets.append(vt)

        ui.button("Add Target", on_click=add_target)
        return inp

    @property
    def target(self):
        if self.target_aggregation is None:
            return ValueTarget(f"{self.base_target.key}")
        else:
            return ValueTarget(f"{self.base_target.key}_{self.target_aggregation}")

    @UIE
    def temporal_avg_target(self, cb): ...

    @UIE
    def base_target(self, cb):
        return ui.select(
            label="Target",
            options=[repr(k) for k in self.sw.value_targets],
            multiple=False,
            on_change=cb,
            value=repr(self.sw.value_targets[0]),
        )

    @base_target.updater
    def base_target(self, e: ui.select):
        e.options = [repr(k) for k in self.sw.value_targets]

    @base_target.value
    def base_target(self, e):
        d = {repr(k): k for k in self.sw.value_targets}
        if e.value not in d:
            return ...
        return d[e.value]

    @UIE
    def heatmap(self, cb):
        return ui.html()

    @heatmap.updater
    def heatmap(self, e):
        if ... in [self.key1, self.key2, self.aggregation, self.cmap, self.base_target]:
            return
        # self.sw.add_target_history()
        self.sw.add_target_averages()
        sa = SweepAnalysis(
            sweep=self.sw, xkeys=self.key1, ykeys=self.key2, target=self.target
        )
        sa.cmap = self.cmap
        e.set_content(
            sa.heatmap(self.aggregation, color_axis=self.color_axis)
            .set_properties(
                **{
                    "text-align": "center",
                    # "border-collapse": "collapse",
                    # "border": "1px solid",
                    "width": "200px",
                }
            )
            .to_html()
        )
        e.update()

    @UIE
    def color_axis(self, cb):
        return ui.select(
            label="Color Axis",
            options=[None, 0, 1],
            multiple=False,
            on_change=cb,
            value=None,
        )

    @UIE
    def key1(self, cb):
        keys = [repr(k) for k in self.sw.keys]
        return ui.select(
            label="Keys1",
            options=keys,
            multiple=True,
            on_change=cb,
            value=keys[0:1],
        )

    @key1.value
    def key1(self, e):
        return SweepKeys([{repr(k): k for k in self.sw.keys}[ev] for ev in e.value])

    @UIE
    def key2(self, cb):
        keys = [repr(k) for k in self.sw.keys]
        return ui.select(
            label="Keys",
            options=keys,
            multiple=True,
            on_change=cb,
            value=keys[-1:],
        )

    @key2.value
    def key2(self, e):
        return SweepKeys([{repr(k): k for k in self.sw.keys}[ev] for ev in e.value])

    @UIE
    def min_run_length(self, cb):
        return ui.input(label="min_run_length", on_change=cb)

    @min_run_length.value
    def min_run_length(self, e):
        return int(e.value)

    @UIE
    def begin_aggregation_phase(self, cb):
        return ui.input(label="Aggregation Begin Step", on_change=cb, value=49000)

    @begin_aggregation_phase.value
    def begin_aggregation_phase(self, e):
        print("agg", e.value)
        try:
            return int(e.value)
        except:
            return 0

    @begin_aggregation_phase.updater
    def begin_aggregation_phase(self, e):
        self.sw.add_target_averages(min_step=self.begin_aggregation_phase)

        # self.ax2 = list(self.skvs.keys)

        # l = self.sw.keys.copy()
        # if self.key1 in l:
        #     l.remove(self.key1)
        # return l

    # @ddmenuprop
    # def a_menu(self):
    #     print("a_menu new value is", self.a_menu)
    #     v = self.a_menu
    #     if v is ...:
    #         return [1, 2, 3]
    #     return [v - 1, v, v + 1]

    @UIE
    def aggregation(self, cb):
        vals = [
            "max",
            "mean",
            "med",
            "min",
        ]
        style_normal = f"background-color: #f0f0f0; border: 4px solid #f0f0f0; border-radius: 8px; padding: 1px; margin: 1px; margin-top: -10px"
        style_selected = f"background-color: #f0f0f0; border: 4px solid #8888f0; border-radius: 8px; padding: 1px; margin: 1px; margin-top: -10px"
        style_clicked = f"background-color: #f0AA77; border: 4px solid #8888f0; border-radius: 8px; padding: 1px; margin: 1px; margin-top: -10px"
        INITIAL = 3
        i = 0
        labels = []
        clicked = INITIAL
        setattr(self, "_selectables", vals[INITIAL])

        def select(i):
            print("selected", vals[i])
            setattr(self, "_selectables", vals[i])
            self.update()

        def mkfns(label, i):
            def hover():
                if clicked != -1:
                    return
                for l in labels:
                    l.style(style_normal)
                label.style(style_selected)
                select(i)
                # l.text = "F" + str(i)

            def click():
                nonlocal clicked
                if clicked == i:
                    clicked = -1
                    hover()
                else:
                    clicked = i
                    for l in labels:
                        l.style(style_normal)
                    label.style(style_clicked)
                    select(i)

            return hover, click

        c = ui.card()
        with c:
            ui.label("Aggregation")
            for i, v in enumerate(vals):
                l = ui.label(v)
                labels.append(l)
                hover, click = mkfns(l, i)
                l.on("mouseover", handler=hover)
                l.on("click", handler=click)
        for i, l in enumerate(labels):
            if i == INITIAL:
                l.style(style_clicked)
            else:
                l.style(style_normal)
        return c

    @aggregation.value
    def aggregation(self, e):
        return self._selectables

    @UIE
    def target_aggregation(self, cb):
        vals = ["max", "mean", "med", "min", "std", None]
        style_normal = f"background-color: #f0f0f0; border: 4px solid #f0f0f0; border-radius: 8px; padding: 1px; margin: 1px; margin-top: -10px"
        style_selected = f"background-color: #f0f0f0; border: 4px solid #8888f0; border-radius: 8px; padding: 1px; margin: 1px; margin-top: -10px"
        style_clicked = f"background-color: #f0AA77; border: 4px solid #8888f0; border-radius: 8px; padding: 1px; margin: 1px; margin-top: -10px"

        i = 0
        INITIAL = 1
        labels = []
        clicked = INITIAL
        setattr(self, "_target_selectables", vals[INITIAL])

        def select(i):
            print("selected", vals[i])
            setattr(self, "_target_selectables", vals[i])
            self.update()

        def mkfns(label, i):
            def hover():
                if clicked != -1:
                    return
                for l in labels:
                    l.style(style_normal)
                label.style(style_selected)
                select(i)
                # l.text = "F" + str(i)

            def click():
                nonlocal clicked
                if clicked == i:
                    clicked = -1
                    hover()
                else:
                    clicked = i
                    for l in labels:
                        l.style(style_normal)
                    label.style(style_clicked)
                    select(i)

            return hover, click

        c = ui.card()
        with c:
            ui.label("Aggregation")
            for i, v in enumerate(vals):
                l = ui.label(v)
                labels.append(l)
                hover, click = mkfns(l, i)
                l.on("mouseover", handler=hover)
                l.on("click", handler=click)
        for i, l in enumerate(labels):
            if i == INITIAL:
                l.style(style_clicked)
            else:
                l.style(style_normal)
        return c

    @target_aggregation.value
    def target_aggregation(self, e):
        return self._target_selectables


class KeyFilters:
    def __init__(self, keys):
        self.keys = keys
        with ui.card():
            ui.label("Filters")
            self.filters = [KeyFilter(k) for k in keys]
            for f in self.filters:
                f.filtering

    @property
    def values(self):
        return [f.filtering for f in self.filters]


class KeyFilter:
    def __init__(self, key: SweepKeys):
        self.key = key
        with ui.row():
            # ui.label(repr(key))
            self.filtering
            self.filter_values

    @UIE
    def filtering(self, cb):
        return ui.checkbox(repr(self.key), on_change=cb)

    @filtering.value
    def filtering(self, e):
        return e.value

    @UIE
    def filter_values(self, cb):
        return ui.select(
            label="Values",
            options=[repr(v) for v in self.key.values],
            multiple=True,
            on_change=cb,
        )

    @filter_values.value
    def filter_values(self, e):
        return [v for v in self.key.values if repr(v) in e.value]


class MetaView:
    ...

    # initial values for newly spawned
    # add new ones
    def __init__(self):
        self.viewcard = ui.card()
        # inputs for new here st they are at end of page
        # + default vals
        # and maybe update all views

    def add(self, sweep):
        with self.viewcard:
            SAView(sweep)


# sv = SAView("sae sweeps/c6ko8r79")


# sv = SAView("sae sweeps/ehwwfzxa")
# SAView("sae sweeps/qib7cabg")
# SAView("sae sweeps/70k6vuhk")
# SAView("sae sweeps/qvb5ec5y")
# SAView("sae sweeps/26luamgd")
# SAView("sae sweeps/wdhl4nju")
# SAView("sae sweeps/js9lfcmn")
# SAView("sae sweeps/5yfl5r4f")
# SAView("sae sweeps/89r31veb")
# SAView("L0Targeting/ye1ap8yb")
SAView("L0Targeting_cmp/vg1mkx3k")
SAView("L0Targeting_cmp/qmamgr4a")

ui.run()
