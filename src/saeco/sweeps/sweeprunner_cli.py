from saeco.sweeps.SweepRunner import SweepRunner
from saeco.sweeps.newsweeper import SweepData
from saeco.mlog import mlog
from pathlib import Path
import click


@click.command()
@click.argument("sweepdata_path", type=click.Path(exists=True))
def start(sweepdata_path: str):
    """Run a sweep using the specified sweep data file.

    Args:
        sweepdata_path: Path to the sweep data file
    """
    sweep_data = SweepData.load(Path(sweepdata_path))
    print("sweep data", sweep_data)
    worker = SweepRunner.from_sweepdata(sweep_data)
    print("made worker")
    worker.start_sweep_agent()


if __name__ == "__main__":
    start()
