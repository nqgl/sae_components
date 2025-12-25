# @dataclass

from datasets import ReadInstruction

from saeco.sweeps import SweepableConfig


class SplitConfig(SweepableConfig):
    split: str = "train"
    start: int | None = None
    end: int | None = None
    tokens_from_split: int | None = None

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

    def get_bounds(self, n: int) -> tuple[int, int]:
        start = int(n * self.start / 100) if self.start is not None else 0
        end = int(n * self.end / 100) if self.end is not None else n
        return start, end
