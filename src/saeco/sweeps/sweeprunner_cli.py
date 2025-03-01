from saeco.sweeps.SweepRunner import SweepRunner
from saeco.sweeps.newsweeper import SweepData
from saeco.mlog import mlog
from pathlib import Path
import click


@click.command()
@click.argument("sweepdata_path", type=click.Path(exists=True))
@click.option("--sweep-index", type=int, default=None)
@click.option("--sweep-hash", type=str, default=None)
def start(sweepdata_path: str, sweep_index: int | None, sweep_hash: str | None):
    """Run a sweep using the specified sweep data file.

    Args:
        sweepdata_path: Path to the sweep data file
    """
    sweep_data = SweepData.load(Path(sweepdata_path))
    worker = SweepRunner.from_sweepdata(
        sweep_data, sweep_index=sweep_index, sweep_hash=sweep_hash
    )
    worker.start_sweep_agent()


if __name__ == "__main__":
    start()
