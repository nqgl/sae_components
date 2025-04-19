from datetime import datetime
from typing import Any, overload

import neptune_scale


@overload
def stringify_unsupported(d: int, parent_key: str = "", sep: str = "/") -> int: ...
@overload
def stringify_unsupported(d: float, parent_key: str = "", sep: str = "/") -> float: ...
@overload
def stringify_unsupported(d: str, parent_key: str = "", sep: str = "/") -> str: ...
@overload
def stringify_unsupported(
    d: datetime, parent_key: str = "", sep: str = "/"
) -> datetime: ...
@overload
def stringify_unsupported(d: bool, parent_key: str = "", sep: str = "/") -> bool: ...
@overload
def stringify_unsupported(
    d: list, parent_key: str = "", sep: str = "/"
) -> dict[str, Any]: ...
@overload
def stringify_unsupported(
    d: set, parent_key: str = "", sep: str = "/"
) -> dict[str, Any]: ...
@overload
def stringify_unsupported(
    d: dict[str, Any], parent_key: str = "", sep: str = "/"
) -> dict[str, Any]: ...
@overload
def stringify_unsupported(
    d: tuple, parent_key: str = "", sep: str = "/"
) -> dict[str, Any]: ...


def stringify_unsupported(d, parent_key: str = "", sep: str = "/"):
    """
    from neptune scale docs, modified for type checker clarity
    """
    SUPPORTED_DATATYPES = [int, float, str, datetime, bool, list, set]

    items = {}
    if not isinstance(d, (dict, list, tuple, set)):
        return d if type(d) in SUPPORTED_DATATYPES else str(d)
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            if isinstance(v, (dict, list, tuple, set)):
                items |= stringify_unsupported(v, new_key, sep=sep)
            else:
                items[new_key] = v if type(v) in SUPPORTED_DATATYPES else str(v)
    elif isinstance(d, (list, tuple, set)):
        for i, v in enumerate(d):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            if isinstance(v, (dict, list, tuple, set)):
                items.update(stringify_unsupported(v, new_key, sep=sep))
            else:
                items[new_key] = v if type(v) in SUPPORTED_DATATYPES else str(v)
    return items


class NeptuneScaleMetricLogger:
    def __init__(self, run: neptune_scale.Run, key: str):
        self.run = run
        self.key = key

    def __getitem__(self, key: str):
        return NeptuneScaleMetricLogger(self.run, f"{self.key}/{key}")

    def __setitem__(self, key: str, value: Any):
        self.run.log_configs(stringify_unsupported({f"{self.key}/{key}": value}))

    def log_metric(self, value: Any, step):
        self.run.log_metrics(
            stringify_unsupported({self.key: value}),
            step=step,
        )
        # self.run.wait_for_processing()
        # print(2)
