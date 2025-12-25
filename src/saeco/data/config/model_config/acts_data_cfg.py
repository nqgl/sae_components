import torch

from saeco.misc.dtypes import str_to_dtype
from saeco.sweeps import SweepableConfig


class ActsDataConfig(SweepableConfig):
    d_data: int = 768
    sites: list[str] = ["transformer.h.6.input"]
    site_d_datas: list[int] | None = None
    excl_first: bool = True
    filter_pad: bool = True
    storage_dtype_str: str | None = None
    autocast_dtype_str: str | bool | None = None
    force_cast_dtype_str: str | None = None

    @property
    def actstring(self):
        sites_str = "_".join(sorted(self.sites))
        return f"{sites_str}_{self.excl_first}_{self.filter_pad}_{self.storage_dtype_str}_{self.autocast_dtype_str}_{self.force_cast_dtype_str}"

    @property
    def storage_dtype(self) -> torch.dtype:
        if self.storage_dtype_str is None:
            return self.force_cast_dtype or self.autocast_dtype or torch.float32
        return str_to_dtype(self.storage_dtype_str)

    @property
    def autocast_dtype(self):
        if self.autocast_dtype_str is False:
            return False
        if self.autocast_dtype_str is None:
            return self.storage_dtype
        return str_to_dtype(self.autocast_dtype_str)

    @property
    def force_cast_dtype(self):
        if self.force_cast_dtype_str is None:
            return None
        return str_to_dtype(self.force_cast_dtype_str)
