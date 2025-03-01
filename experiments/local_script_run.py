from saeco.sweeps.SweepRunner import SweepRunner
from saeco.sweeps.newsweeper import SweepData
from pathlib import Path


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


start("sweeprefs/None/cyujlaxg.sweepdata")
