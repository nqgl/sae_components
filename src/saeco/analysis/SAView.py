from saeco.analysis.ddmenuprop import ddmenuprop, ddupdate
from saeco.analysis.wandb_analyze import Sweep, SweepAnalysis, SweepKeys, ValueTarget
from saeco.analysis.uiitem import UIE

from nicegui import ui
import asyncio


class SAView:
    def __init__(self, sweep="sae sweeps/5uwxiq76"):
        self.sw = Sweep(sweep)
        self.sa = None
        with ui.row():
            with ui.card() as c:
                # label = ui.label(f"Sweep {sweep}")
                # with ui.card():
                #     label = ui.label(f"Keys:")
                # with ui.card():
                #     label = ui.label(f"Values:")
                self.heatmap
                with ui.row():
                    self.aggregation
                    with ui.card():
                        self.key1
                        self.key2

                self.cmap = "viridis"
            with ui.row():
                with ui.card():
                    # ui.label("Target Selection")
                    # self.aggregation
                    self.base_target
                    self.new_value_target
                    self.begin_aggregation_phase
                self.target_aggregation
                self.update_hist

    @UIE
    def update_hist(self, cb):
        return ui.button("Update", on_click=self.sw.add_target_history_async)

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
        e.set_content(
            sa.heatmap(self.aggregation)
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
    def key1(self, cb):
        return ui.select(
            label="Keys",
            options=[repr(k) for k in self.sw.keys],
            multiple=True,
            on_change=cb,
        )

    @key1.value
    def key1(self, e):
        return SweepKeys([{repr(k): k for k in self.sw.keys}[ev] for ev in e.value])

    @UIE
    def key2(self, cb):
        return ui.select(
            label="Keys",
            options=[repr(k) for k in self.sw.keys],
            multiple=True,
            on_change=cb,
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
        return ui.input(label="Aggregation Begin Step", on_change=cb)

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
        style_normal = f"background-color: #f0f0f0; border: 5px solid #f0f0f0; border-radius: 5px; padding: 1px; margin: 3px; margin-top: -10px"
        style_selected = f"background-color: #f0AA77; border: 5px solid #f0f0f0; border-radius: 5px; padding: 1px; margin: 3px; margin-top: -10px"
        style_clicked = f"background-color: #C03399; border: 5px solid #f0f0f0; border-radius: 5px; padding: 1px; margin: 3px; margin-top: -10px"

        i = 0
        labels = []
        clicked = -1
        setattr(self, "_selectables", None)

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
            for i, v in enumerate(vals):
                l = ui.label(v)
                labels.append(l)
                hover, click = mkfns(l, i)
                l.on("mouseover", handler=hover)
                l.on("click", handler=click)
        for l in labels:
            l.style(style_normal)
        return c

    @aggregation.value
    def aggregation(self, e):
        return self._selectables

    @UIE
    def target_aggregation(self, cb):
        vals = ["max", "mean", "med", "min", "std", None]
        style_normal = f"background-color: #f0f0f0; border: 5px solid #f0f0f0; border-radius: 5px; padding: 1px; margin: 3px; margin-top: -10px"
        style_selected = f"background-color: #f0AA77; border: 5px solid #f0f0f0; border-radius: 5px; padding: 1px; margin: 3px; margin-top: -10px"
        style_clicked = f"background-color: #C03399; border: 5px solid #f0f0f0; border-radius: 5px; padding: 1px; margin: 3px; margin-top: -10px"

        i = 0
        labels = []
        clicked = -1
        setattr(self, "_target_selectables", None)

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
            for i, v in enumerate(vals):
                l = ui.label(v)
                labels.append(l)
                hover, click = mkfns(l, i)
                l.on("mouseover", handler=hover)
                l.on("click", handler=click)
        for l in labels:
            l.style(style_normal)
        return c

    @target_aggregation.value
    def target_aggregation(self, e):
        return self._target_selectables


# sv = SAView("sae sweeps/c6ko8r79")
sv = SAView("sae sweeps/z0dm6lcf")


ui.run()
