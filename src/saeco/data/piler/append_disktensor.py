import tables
import torch
from safetensors.torch import load_file, save_file


from pathlib import Path
from typing import List, Union

dtype_to_atom = {
    torch.float32: tables.Float32Atom(),
    torch.int32: tables.Int32Atom(),
    torch.int64: tables.Int64Atom(),
    torch.float16: tables.Float16Atom(),
}


class AppendDiskTensor:
    def __init__(
        self,
        path: Union[str, Path],
        dtype: torch.dtype,
        fixed_shape: List[int],
    ):
        if isinstance(path, str):
            path = Path(path)
        self.path: Path = path
        self.dtype = dtype
        self.fixed_shape = fixed_shape

    def init_file(self, force=False):
        assert force or not self.path.exists()
        table = tables.open_file(str(self.path), mode="w")
        table.create_earray(
            table.root,
            "batches",
            atom=dtype_to_atom[self.dtype],
            shape=(0, *self.fixed_shape),
        )
        table.close()

    def write(self, t: torch.Tensor):
        if not self.path.exists():
            self.init_file()
        assert t.dtype == self.dtype, (t.dtype, self.dtype)
        assert t.shape[1:] == torch.Size(self.fixed_shape), (t.shape, self.fixed_shape)
        table = tables.open_file(str(self.path), mode="a")
        table.root.batches.append(t.cpu().numpy())
        table.close()

    def read(self):
        table = tables.open_file(str(self.path), mode="r")
        t = torch.tensor(table.root.batches[:])
        table.close()
        return t

    def shuffle(self):
        t = self.read()
        t = t[torch.randperm(t.shape[0])]
        self.init_file(force=True)
        self.write(t)

    def shuffle_and_finalize_pt_old(self):
        """
        shuffle the .h5, turn it into a tensor saved as .pt, then deletes the original .h5
        """
        t = self.read()
        t = t[torch.randperm(t.shape[0])]
        torch.save(t, str(self.path).split(".")[0] + ".pt")
        self.path.unlink()

    def shuffle_and_finalize(self):
        """
        shuffle the .h5, turn it into a tensor saved as .safetensors, then deletes the original .h5
        """
        pt = self.path.with_suffix(".pt")

        if pt.exists():
            t = self.readtensor(pt=True, mmap=True)
        else:
            t = self.read()
            t = t[torch.randperm(t.shape[0])]
        save_file({"tensor": t}, self.path.with_suffix(".safetensors"))
        self.path.unlink(missing_ok=True)

    def readtensor(self, pt=False, mmap=False):
        if self.path.exists():
            self.shuffle_and_finalize()
        ptpath = self.path.with_suffix(".pt")

        if pt:
            if ptpath.exists():
                return torch.load(
                    str(self.path).split(".")[0] + ".pt",
                    map_location=(
                        torch.device("cpu")
                        if self.dtype == torch.float16
                        else torch.device("cpu")
                    ),
                    mmap=mmap,
                    weights_only=True,
                )
        if ptpath.exists():
            print(f"converting .pt {ptpath} to safetensors")
            self.shuffle_and_finalize()
            ptpath.unlink()
        st = self.path.with_suffix(".safetensors")
        return load_file(st)["tensor"]
