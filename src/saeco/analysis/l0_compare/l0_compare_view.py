from nicegui import ui
import plotly.graph_objects as go
from saeco.analysis.uiitem import UIE
from saeco.analysis.wandb_analyze import Sweep, ValueTarget
import numpy as np


class L0CompareView:
    def __init__(self, baseline_sweep_path: str, targeted_sweep_path: str):
        self.baseline_sweep = Sweep(baseline_sweep_path)
        self.targeted_sweep = Sweep(targeted_sweep_path)
        self.fig = None
        # Initialize UI components
        with ui.card():
            ui.label("L0 Targeting Comparison")
            with ui.row():
                with ui.card():
                    ui.label("Plot Settings")
                    self.x_axis_target
                    self.y_axis_target
                    with ui.card():
                        ui.label("Aggregation Settings")
                        self.aggregation_type
                        self.aggregation_step
                    self.update_plot_button
                    self.update_cb

                with ui.card():
                    with ui.label("Plot"):
                        ui.separator()
                        # self.plot_card
                        self.plot
        self.targeted_sweep._df = None
        self.baseline_sweep._df = None

        self.hist_update(self.baseline_sweep)
        self.hist_update(self.targeted_sweep)

    # @UIE
    # def plot_card(self, cb):
    #     e = ui.card()
    #     with e:
    #         initial_fig = go.Figure()
    #         initial_fig.update_layout(
    #             title="L0 Targeting Comparison",
    #             xaxis_title="Select X-Axis Metric",
    #             yaxis_title="Select Y-Axis Metric",
    #             template="plotly_white",
    #             width=800,
    #             height=600,
    #         )
    #         ui.plotly(initial_fig)

    #     return e

    @UIE
    def plot(self, cb):
        initial_fig = go.Figure()
        initial_fig.update_layout(
            title="L0 Targeting Comparison",
            xaxis_title="Select X-Axis Metric",
            yaxis_title="Select Y-Axis Metric",
            template="plotly_white",
            width=800,
            height=600,
        )
        return ui.plotly(initial_fig)

    @plot.updater
    def plot(self, e: ui.plotly):
        if self.fig is not None:
            e.update_figure(self.fig)

    # @plot_card.updater
    # def plot_card(self, e):
    #     print("Updating plot")
    #     e.clear()  # Clear existing content
    #     with e:
    #         if self.fig is not None:
    #             ui.plotly(self.fig)
    #         else:
    #             initial_fig = go.Figure()
    #             initial_fig.update_layout(
    #                 title="L0 Targeting Comparison",
    #                 xaxis_title="Select X-Axis Metric",
    #                 yaxis_title="Select Y-Axis Metric",
    #                 template="plotly_white",
    #                 width=800,
    #                 height=600,
    #             )
    #             ui.plotly(initial_fig)

    @UIE
    def x_axis_target(self, cb):
        return ui.select(
            label="X-Axis Metric",
            options=[t.nicename for t in self.baseline_sweep.value_targets],
            on_change=cb,
        )

    @UIE
    def y_axis_target(self, cb):
        return ui.select(
            label="Y-Axis Metric",
            options=[t.nicename for t in self.baseline_sweep.value_targets],
            on_change=cb,
        )

    @UIE
    def aggregation_type(self, cb):
        return ui.select(
            label="Aggregation Type",
            options=["mean", "median", "min", "max"],
            value="mean",
            on_change=cb,
        )

    @UIE
    def aggregation_step(self, cb):
        return ui.number(label="Aggregation Start Step", value=49000, on_change=cb)

    # @aggregation_step.updater
    # def aggregation_step(self, e):

    @UIE
    def update_plot_button(self, cb):
        def on_click():
            self.targeted_sweep.add_target_averages(
                min_step=self.aggregation_step, force=True
            )
            self.baseline_sweep.add_target_averages(
                min_step=self.aggregation_step, force=True
            )
            self.update_plot()

            cb()
            cb()

        return ui.button("Update Plot", on_click=on_click)

    @UIE
    def update_cb(self, cb):
        return ui.button("Call Callback", on_click=cb)

    def hist_update(self, sweep: Sweep):
        sweep.add_target_history()

        if self.aggregation_step is not None:
            sweep.add_target_averages(min_step=self.aggregation_step, force=True)

    def get_target_values(self, sweep: Sweep, target_name: str) -> list[float]:
        """Get values for a specific target from a sweep's DataFrame"""
        # Make sure we have the latest target averages with current aggregation step
        # self.hist_update(sweep)
        # Get the aggregated values based on selected type
        agg_key = f"{target_name}_{self.aggregation_type}"
        if self.aggregation_type == "median":
            agg_key = f"{target_name}_med"  # Handle special case for median

        if agg_key in sweep.df.columns:
            return sweep.df[agg_key].tolist()

        # Fallback to raw values if aggregation not available
        if target_name in sweep.df.columns:
            return sweep.df[target_name].tolist()

        return []

    # def _create_plot(self):
    #     """Create a new plot with current data"""
    #     with self.plot_card:
    #         self.plot_card.clear()  # Clear existing content
    #         initial_fig = go.Figure()
    #         initial_fig.update_layout(
    #             title="L0 Targeting Comparison",
    #             xaxis_title="Select X-Axis Metric",
    #             yaxis_title="Select Y-Axis Metric",
    #             template="plotly_white",
    #             width=800,
    #             height=600,
    #         )
    #         self.plot_container = ui.plotly(initial_fig)

    def update_plot(self):

        if not self.x_axis_target or not self.y_axis_target:
            return

        # Get values for both sweeps
        baseline_x = np.array(
            self.get_target_values(self.baseline_sweep, self.x_axis_target)
        )
        baseline_y = np.array(
            self.get_target_values(self.baseline_sweep, self.y_axis_target)
        )
        targeted_x = np.array(
            self.get_target_values(self.targeted_sweep, self.x_axis_target)
        )
        targeted_y = np.array(
            self.get_target_values(self.targeted_sweep, self.y_axis_target)
        )

        # Create the Plotly figure
        fig = go.Figure()

        # Add baseline points
        fig.add_trace(
            go.Scatter(
                x=baseline_x,
                y=baseline_y,
                mode="markers",
                name="Baseline",
                marker=dict(size=10, opacity=0.6),
                hovertemplate=f"{self.x_axis_target}: %{{x}}<br>"
                + f"{self.y_axis_target}: %{{y}}<br>"
                + "<extra>Baseline</extra>",
            )
        )

        # Add targeted points
        fig.add_trace(
            go.Scatter(
                x=targeted_x,
                y=targeted_y,
                mode="markers",
                name="Targeted",
                marker=dict(size=10, opacity=0.6),
                hovertemplate=f"{self.x_axis_target}: %{{x}}<br>"
                + f"{self.y_axis_target}: %{{y}}<br>"
                + "<extra>Targeted</extra>",
            )
        )

        # Update layout
        fig.update_layout(
            title="L0 Targeting Comparison",
            xaxis_title=self.x_axis_target,
            yaxis_title=self.y_axis_target,
            hovermode="closest",
            template="plotly_white",
            width=800,
            height=600,
        )

        self.fig = fig
        # self.update()
        # return fig
        self.update()

    def update(self):
        pass


def get_close_pairs(l1: list[float], l2: list[float]) -> list[tuple[int, int]]:
    """Returns pairs of indices in l1 and l2 that are closest"""
    l1se = sorted(list(enumerate(l1)), key=lambda x: x[1])
    l2se = sorted(list(enumerate(l2)), key=lambda x: x[1])
    return [(l1se[i][0], l2se[i][0]) for i in range(min(len(l1), len(l2)))]


baseline_runs_name = "L0Targeting_cmp/f5jbxrmd"
targeted_runs_name = "L0Targeting_cmp/zvvtvwt0"
new_baseline_runs_name = "L0Targeting_cmp/dj9x0ne2"
new_targeted_runs_name = "L0Targeting_cmp/umjlb4dt"
L0CompareView(new_baseline_runs_name, new_targeted_runs_name)
ui.run()
