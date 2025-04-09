from saeco.sweeps.SweepRunner import SweepRunner
from saeco.sweeps.newsweeper import SweepData
from saeco.mlog import mlog
from pathlib import Path
import click


@click.command()
@click.argument("sweepdata_path", type=click.Path())
@click.option("--sweep-index", type=int, default=None)
@click.option("--sweep-hash", type=str, default=None)
@click.option(
    "--distributed-skip-log", type=bool, default=False, is_flag=True, flag_value=True
)
def start(
    sweepdata_path: str,
    sweep_index: int | None,
    sweep_hash: str | None,
    distributed_skip_log: bool,
):
    """Run a sweep using the specified sweep data file.

    Args:
        sweepdata_path: Path to the sweep data file
    """
    print("cwd", Path.cwd())
    try:
        path = Path.cwd() / sweepdata_path
    except ValueError:
        path = Path(sweepdata_path)
    sweep_data = SweepData.load(path)
    worker = SweepRunner.from_sweepdata(
        sweep_data, sweep_index=sweep_index, sweep_hash=sweep_hash
    )
    if distributed_skip_log:
        from composer.utils import dist

        assert dist.get_world_size() > 1
        if dist.get_global_rank() != 0:
            mlog.start_sweep_agent(
                sweep_data,
                worker.run_sweep_dont_enter,
                sweep_index=sweep_index,
                sweep_hash=sweep_hash,
            )
            return
    worker.start_sweep_agent()


if __name__ == "__main__":
    start()
