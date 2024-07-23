#!/usr/bin/env python3
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

import numpy as np
from matplotlib import pyplot as plt
from nicegui import ui
from saeco.analysis.ddmenuprop import ddmenuprop, ddupdate

# from saeco.analysis.update_render import update_render, render_update_list
from saeco.analysis.wandb_analyze import (
    Key,
    ValueTarget,
    SweepKey,
    SweepKeys,
    SetKeys,
    Sweep,
    SweepAnalysis,
)

# df = pd.DataFrame(
#     data={
#         "col1": [x for x in range(4)],
#         "col2": ["This", "column", "contains", "strings."],
#         "col3": [x / 4 for x in range(4)],
#         "col4": [True, False, True, False],
#     }
# )


# def update(*, df: pd.DataFrame, r: int, c: int, value):
#     df.iat[r, c] = value
#     ui.notify(f"Set ({r}, {c}) to {value}")


# with ui.grid(rows=len(df.index) + 1).classes("grid-flow-col"):
#     for c, col in enumerate(df.columns):
#         ui.label(col).classes("font-bold")
#         for r, row in enumerate(df.loc[:, col]):
#             if is_bool_dtype(df[col].dtype):
#                 cls = ui.checkbox
#             elif is_numeric_dtype(df[col].dtype):
#                 cls = ui.number
#             else:
#                 cls = ui.input
#             cls(
#                 value=row,
#                 on_change=lambda event, r=r, c=c: update(
#                     df=df, r=r, c=c, value=event.value
#                 ),
#             )


class ExObj:
    def __init__(self):
        self.field = 7

    def value(self):
        return self.field * 2


# o = ExObj()
# label = ui.label(f"Field: {o.field}, Value: {o.value()}")


# def render():
#     label.text = f"Field: {o.field}, Value: {o.value()}"


def make_setfield_menu(obj, field, options):
    def setvalue(value, b):
        def setter():
            setattr(obj, field, value)
            b.text = f"{field}={value}"
            b.update()
            ui.update()

        return setter

    with ui.dropdown_button(f"{field}", auto_close=True) as b:
        for opt in options:
            ui.item(repr(opt), on_click=setvalue(opt, b))

        # ui.item("Item 2", on_click=lambda: ui.notify("You clicked item 2"))

    # return wrap(optfn)
    # return wrap


from nicegui import ui
import matplotlib.pyplot as plt

# with ui.pyplot(figsize=(3, 2)):
#     x = np.linspace(0.0, 5.0)
#     y = np.cos(2 * np.pi * x) * np.exp(-x)
#     plt.plot(sa.heatmap("max"))
#     # plt.plot(x, y, "-")

ui.run()


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


# make_setfield_menu(o, "field", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
saview = SAView("sae sweeps/mfwai3n2")
saview2 = SAView()

ui.run()
