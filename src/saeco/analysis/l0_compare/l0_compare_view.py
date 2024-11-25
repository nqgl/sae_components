from nicegui import ui
import matplotlib.pyplot as plt
from saeco.analysis.uiitem import UIE
from saeco.analysis.wandb_analyze import Sweep, ValueTarget
import numpy as np


class L0CompareView:
    def __init__(self, baseline_sweep_path: str, targeted_sweep_path: str):
        self.baseline_sweep = Sweep(baseline_sweep_path)
        self.targeted_sweep = Sweep(targeted_sweep_path)
        self.targeted_sweep._df = None
        self.baseline_sweep._df = None
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

                with ui.card():
                    with ui.label("Plot"):
                        ui.separator()
                        self.plot_container = (
                            ui.html()
                        )  # Container for matplotlib plot as HTML

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

    @UIE
    def update_plot_button(self, cb):
        return ui.button("Update Plot", on_click=self.update_plot)

    def hist_update(self, sweep: Sweep):
        sweep.add_target_history()

        if self.aggregation_step is not None:
            sweep.add_target_averages(min_step=self.aggregation_step, force=True)

    def get_target_values(self, sweep: Sweep, target_name: str) -> list[float]:
        """Get values for a specific target from a sweep's DataFrame"""
        # Make sure we have the latest target averages with current aggregation step
        self.hist_update(sweep)
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

    def update_plot(self):
        self.targeted_sweep._df = None
        self.baseline_sweep._df = None

        if not self.x_axis_target or not self.y_axis_target:
            return

        # Get values for both sweeps
        baseline_x = self.get_target_values(self.baseline_sweep, self.x_axis_target)
        baseline_y = self.get_target_values(self.baseline_sweep, self.y_axis_target)
        targeted_x = self.get_target_values(self.targeted_sweep, self.x_axis_target)
        targeted_y = self.get_target_values(self.targeted_sweep, self.y_axis_target)

        # Create pairs of indices for matching runs
        pairs = get_close_pairs(baseline_x, targeted_x)

        # Create the plot
        plt.figure(figsize=(10, 6))

        # Plot baseline points
        plt.scatter(baseline_x, baseline_y, label="Baseline", alpha=0.6)
        # Plot targeted points
        plt.scatter(targeted_x, targeted_y, label="Targeted", alpha=0.6)

        # Draw lines connecting paired points
        for baseline_idx, targeted_idx in pairs:
            plt.plot(
                [baseline_x[baseline_idx], targeted_x[targeted_idx]],
                [baseline_y[baseline_idx], targeted_y[targeted_idx]],
                "k-",
                alpha=0.2,
            )

        plt.xlabel(self.x_axis_target)
        plt.ylabel(self.y_axis_target)
        plt.title("L0 Targeting Comparison")
        plt.legend()

        # Convert plot to HTML and update the container
        import io
        import base64

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")

        # Update the HTML content with the image
        self.plot_container.set_content(
            f'<img src="data:image/png;base64,{img_str}" style="width:100%; max-width:800px;">'
        )
        self.update()
        plt.close()

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
new_targeted_runs_name = "L0Targeting_cmp/19zx1k97"
L0CompareView(new_baseline_runs_name, new_targeted_runs_name)
ui.run()
