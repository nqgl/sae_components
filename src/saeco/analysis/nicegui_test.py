#!/usr/bin/env python3
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

import numpy as np
from matplotlib import pyplot as plt
from nicegui import ui
from saeco.analysis.SAView import SAView

# from saeco.analysis.update_render import update_render, render_update_list
from saeco.analysis.wandb_analyze import (
    Key,
    ValueTarget,
    SweepKey,
    SetKeys,
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


# make_setfield_menu(o, "field", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
saview = SAView("sae sweeps/mfwai3n2")
saview2 = SAView()

ui.run()
