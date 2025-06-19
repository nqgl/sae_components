# @dataclass
from typing import Optional

from datasets import ReadInstruction

from saeco.sweeps import SweepableConfig


class SplitConfig(SweepableConfig):
    split: str = "train"
    start: int | None = None
    end: int | None = None
    act_chunks_cached: int | None = None
    acts_per_chunk: int | None = None

    @property
    def split_dir_id(self):
        if self.start is None and self.end is None:
            return self.split
        return f"{self.split}[{self.start}_p:{self.end}_p]"

    def get_split_key(self):
        """
        Value passed to the "split" keyword argument of datasets.load_dataset
        """
        if self.start is None and self.end is None:
            return self.split
        return ReadInstruction(
            self.split,
            from_=self.start,
            to=self.end,
            unit="%",
        )
