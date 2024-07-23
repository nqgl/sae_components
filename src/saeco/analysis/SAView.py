from saeco.analysis.ddmenuprop import ddmenuprop, ddupdate
from saeco.analysis.wandb_analyze import Sweep, SweepAnalysis


from nicegui import ui


class SAView:
    def __init__(self, sweep="sae sweeps/5uwxiq76"):
        self.sw = Sweep(sweep)

        sks = self.sw.keys[1] * self.sw.keys[2]
        self.sa = SweepAnalysis(self.sw, sks)
        # with ui.header():
        #     label = ui.label("h label")
        with ui.card() as c:
            label = ui.label(f"Sweep {sweep}")
            with ui.card():
                label = ui.label(f"Keys:")
            with ui.card():
                label = ui.label(f"Values:")
            self.heatmap = ui.html()
            # make_setfield_menu(self, "aggregation", ["mean", "min", "max", "med"])
            self.a_menu
            with ui.row():
                self.aggregation
                self.key1
                self.key2

            self.cmap = "viridis"

            def setcolor(e):
                self.cmap = e.value

            color = ui.input(label="Color", value="viridis", on_change=setcolor)
        ddupdate()

        # render_update_list.append(self.update)

    def update(self):
        sa = SweepAnalysis(self.sw, self.key1 * self.key2)
        sa.cmap = self.cmap
        self.heatmap.set_content(
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
        self.heatmap.update()
        ddupdate()

    @ddmenuprop
    def aggregation(self):
        return ["mean", "min", "max", "med"]

    @ddmenuprop
    def key1(self):
        l = self.sw.keys.copy()
        if self.key2 in l:
            l.remove(self.key2)
        return l

    @ddmenuprop
    def key2(self):
        l = self.sw.keys.copy()
        if self.key1 in l:
            l.remove(self.key1)
        return l

    @ddmenuprop
    def a_menu(self):
        print("a_menu new value is", self.a_menu)
        v = self.a_menu
        if v is ...:
            return [1, 2, 3]
        return [v - 1, v, v + 1]
