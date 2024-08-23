# @dataclass
from saeco.sweeps import SweepableConfig
from typing import Optional


class SplitConfig(SweepableConfig):
    splitname: str
    start: int
    end: int
    split_key: str = "train"
    tokens_from_split: Optional[int] = None

    @property
    def split_dir_id(self):
        return f"{self.split_key}[{self.start}_p:{self.end}_p]"

    def get_split_key(self):
        return f"{self.split_key}[{self.start}%:{self.end}%]"
