from attr import define, field
from .storage.growing_disk_tensor import GrowingDiskTensor
from .storage.disk_tensor import DiskTensor

@define
class MetaData:
    @classmethod
    def create(cls, size, dtype, doc/seq)


