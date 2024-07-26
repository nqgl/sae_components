from saeco.analysis.ddmenuprop import ddmenuprop, ddupdate
from saeco.analysis.wandb_analyze import Sweep, SweepAnalysis, SweepKeys
from saeco.analysis.uiitem import UIE

from nicegui import ui


class SAView:
    def __init__(self, sweep="sae sweeps/5uwxiq76"):
        self.sw = Sweep(sweep)

        # sks =
        self.sa = None
        # with ui.header():
        #     label = ui.label("h label")
        with ui.card() as c:
            label = ui.label(f"Sweep {sweep}")
            with ui.card():
                label = ui.label(f"Keys:")
            with ui.card():
                label = ui.label(f"Values:")
            # self.heatmap = ui.html()
            self.heatmap
            # make_setfield_menu(self, "aggregation", ["mean", "min", "max", "med"])
            with ui.row():
                self.aggregation

                self.key1
                self.key2

            self.cmap = "viridis"

            def setcolor(e):
                self.cmap = e.value

            color = ui.input(label="Color", value="viridis", on_change=setcolor)
        # ddupdate()
        self.aggregation

        # render_update_list.append(self.update)

    def update(self):
        sa = SweepAnalysis(self.sw, self.key1, self.key2)
        sa.cmap = self.cmap
        ddupdate()

    @UIE
    def heatmap(self, cb):
        return ui.html()

    @heatmap.updater
    def heatmap(self, e):
        if ... in [self.key1, self.key2, self.aggregation, self.cmap]:
            return
        sa = SweepAnalysis(self.sw, self.key1, self.key2)
        sa.cmap = self.cmap
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

    # @ddmenuprop
    # def aggregation(self):
    #     return ["mean", "min", "max", "med"]

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
        # l = self.sw.keys.copy()
        # if self.key2 in l:
        #     l.remove(self.key2)
        # return l

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


sv = SAView("sae sweeps/c6ko8r79")
ui.run()
