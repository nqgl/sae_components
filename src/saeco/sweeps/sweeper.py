import argparse
import importlib
from pathlib import Path

parser = argparse.ArgumentParser(description="Sweeper for Saeco")

# parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument("--sweep-folder-path", type=str, default="sweep")
parser.add_argument("--init", action="store_true")


sweepfile = importlib.import_module(parser.sweepfile)


class Sweeper:
    def __init__(self, path):
        self.path = Path(path)
        self.sweepfile = importlib.import_module(self.path / "sweepfile.py")
