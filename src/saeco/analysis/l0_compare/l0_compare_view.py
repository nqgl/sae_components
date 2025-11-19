import pandas as pd
import plotly.graph_objects as go
from nicegui import ui

from saeco.analysis.SAView import KeyFilters
from saeco.analysis.uiitem import UIE
from saeco.analysis.wandb_analyze import Sweep, SweepKeys, ValueTarget

VALUE_TARGETS = [
    ValueTarget("cache/L0"),
    ValueTarget("cache/L2_loss"),
    ValueTarget("cache/L1"),
    ValueTarget("eval/L0"),
    ValueTarget("eval/L2_loss"),
    ValueTarget("eval/L1"),
    ValueTarget("recons/no_bos/nats_lost"),
    ValueTarget("recons/with_bos/nats_lost"),
]


class SweepCompareConfigurationPanel:
    def __init__(self, sweep: Sweep | str, name: str | None = None):
        self.sweep = sweep if isinstance(sweep, Sweep) else Sweep(sweep)
        self.name = name or sweep if isinstance(sweep, str) else sweep.sweep_path
        self.id = self.sweep.sweep_path

        # Initialize UI components
        with ui.card():
            ui.label(f"Sweep: {self.name}")
            ui.label(f"ID: {self.id}")
            ui.separator()

            # Category selection (what would have been rows/columns in heatmap)
            with ui.card():
                ui.label("Categories")
                self.category_keys

            # Filtering
            with ui.card():
                ui.label("Filters")
                self.filters = KeyFilters(self.sweep.keys)

    @UIE
    def category_keys(self, cb):
        """Keys that will create separate categories in the plot legend"""
        keys = [repr(k) for k in self.sweep.keys]
        return ui.select(
            label="Category Keys",
            options=keys,
            multiple=True,
            on_change=cb,
            value=[],  # Default to no categories
        )

    @category_keys.value
    def category_keys(self, e):
        return SweepKeys([{repr(k): k for k in self.sweep.keys}[ev] for ev in e.value])

    def get_categories(self) -> dict[str, pd.DataFrame]:
        """Get data frames for each category based on selected category keys"""
        # Apply filters to the base DataFrame
        filtered_df = self.sweep.df.copy()
        for filter_obj, filter_val in zip(self.filters.filters, self.filters.values):
            if filter_val:  # If the filter is enabled
                key = filter_obj.key.key
                values = filter_obj.filter_values
                if values:  # If specific values are selected
                    filtered_df = filtered_df[filtered_df[key].isin(values)]

        if len(self.category_keys.keys) == 0:
            # If no categories selected, return all data as one category
            return {self.name: filtered_df}

        categories = {}
        for keys in self.category_keys:
            # Get the actual keys from the SetKeys object if it is one
            key_list = list(keys.d.keys()) if hasattr(keys, "d") else list(keys)
            key_names = [k.key for k in key_list]
            # Group data by the selected keys
            for key_values, group_df in filtered_df.groupby(key_names):
                if not isinstance(key_values, tuple):
                    key_values = (key_values,)
                category_name = f"{self.name} ({', '.join(map(str, key_values))})"
                categories[category_name] = group_df

        return categories


class L0CompareView:
    def __init__(self):
        self.sweep_panels: dict[
            str, SweepCompareConfigurationPanel
        ] = {}  # name -> SweepCompareConfigurationPanel
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
                    with ui.card():
                        ui.label("Line Drawing Settings")
                        self.draw_lines
                        self.line_category
                        self.sort_metric
                    self.update_plot_button
                    self.update_cb

                with ui.card():
                    with ui.label("Plot"):
                        ui.separator()
                        self.plot

            # Sweep configuration panels
            with ui.card():
                ui.label("Sweep Configurations")
                self.sweep_cfg_panels

    def add_sweep(self, sweep: Sweep | str, name: str | None = None):
        """Add a new sweep to compare"""
        panel = SweepCompareConfigurationPanel(sweep, name)
        self.sweep_panels[panel.name] = panel
        return panel

    @UIE
    def sweep_cfg_panels(self, cb):
        tabs = ui.tabs()
        with tabs:
            for name, panel in self.sweep_panels.items():
                with ui.tab(name):
                    panel  # This will display the panel's UI

        return tabs

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

    @UIE
    def x_axis_target(self, cb):
        return ui.select(
            label="X-Axis Metric",
            options=[t.nicename for t in VALUE_TARGETS],
            on_change=cb,
            value=[t.nicename for t in VALUE_TARGETS][0],
        )

    @UIE
    def y_axis_target(self, cb):
        return ui.select(
            label="Y-Axis Metric",
            options=[t.nicename for t in VALUE_TARGETS],
            on_change=cb,
            value=[t.nicename for t in VALUE_TARGETS][1],
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
    def draw_lines(self, cb):
        return ui.checkbox("Draw Lines Between Points", on_change=cb)

    @UIE
    def line_category(self, cb):
        return ui.select(
            label="Draw Lines For Category",
            options=["None"] + [name for name in self.sweep_panels.keys()],
            on_change=cb,
            value="None",
        )

    @line_category.updater
    def line_category(self, e: ui.select):
        e.set_options(["None"] + [name for name in self.sweep_panels.keys()])

    @UIE
    def sort_metric(self, cb):
        return ui.select(
            label="Sort Points By",
            options=[t.nicename for t in VALUE_TARGETS],
            on_change=cb,
            value=[t.nicename for t in VALUE_TARGETS][0],
        )

    @UIE
    def update_plot_button(self, cb):
        def on_click():
            for panel in self.sweep_panels.values():
                if "history" not in panel.sweep.df.columns:
                    panel.sweep.add_target_history()
                panel.sweep.add_target_averages(
                    min_step=self.aggregation_step, force=True
                )
            self.update_plot()
            cb()
            cb()

        return ui.button("Update Plot", on_click=on_click)

    @UIE
    def update_cb(self, cb):
        return ui.button("Call Callback", on_click=cb)

    def update_plot(self):
        if not self.x_axis_target or not self.y_axis_target:
            return

        # Create the Plotly figure
        fig = go.Figure()

        # Add traces for each sweep and its categories
        for panel in self.sweep_panels.values():
            categories = panel.get_categories()

            for category_name, category_df in categories.items():
                # Sort points if line drawing is enabled
                if self.draw_lines and category_name == self.line_category:
                    category_df = category_df.sort_values(by=self.sort_metric)

                x_values = category_df[self.x_axis_target].tolist()
                y_values = category_df[self.y_axis_target].tolist()

                # Determine plotting mode based on line drawing setting
                mode = (
                    "markers+lines"
                    if (self.draw_lines and category_name == self.line_category)
                    else "markers"
                )

                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=y_values,
                        mode=mode,
                        name=category_name,
                        marker=dict(size=10, opacity=0.6),
                        line=(
                            dict(dash="dot")
                            if (self.draw_lines and category_name == self.line_category)
                            else None
                        ),
                        hovertemplate=(
                            f"{self.x_axis_target}: %{{x}}<br>"
                            f"{self.y_axis_target}: %{{y}}<br>"
                            f"{self.sort_metric}: %{{customdata}}<br>"
                            f"<extra>{category_name}</extra>"
                        ),
                        customdata=category_df[self.sort_metric].tolist(),
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
        self.update()

    def update(self):
        pass


# Example usage:
# if __name__ == "__main__":
view = L0CompareView()
view.add_sweep("L0Targeting_cmp/dj9x0ne2", "Baseline")
view.add_sweep("L0Targeting_cmp/umjlb4dt", "Targeted")
view.add_sweep("L0Targeting_cmp/wt5a9w7t", "plateaued 5k")
view.add_sweep("L0Targeting_cmp/ldpwjpll", "deflated 0.2")
view.add_sweep("L0Targeting_cmp/sweeps/rx1w98jh", "hiatus test")
view.add_sweep("L0Targeting_cmp/sweeps/azete078", "new")
view.add_sweep("L0Targeting_cmp/sweeps/rqld0e1g", "new2")
ui.run()
