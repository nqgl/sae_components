import torch

from saeco.evaluation import Evaluation

STORAGE_NAME = "stored_acts"
root = Evaluation.from_cache_name(STORAGE_NAME)


def add_metadata(n):
    name = f"mod{n}"
    if name not in root.metadatas:
        b = root.metadata_builder(torch.long, "cpu")
        for chunk in b:
            clen = chunk.tokens.value.shape[0]
            t = torch.arange(n).repeat(clen // n + 1)
            b << t[:clen]
        root.metadatas[name] = b.value
        print(f"added {name}")
    else:
        print(f"{name} already exists")


add_metadata(3)
add_metadata(4)
add_metadata(5)
add_metadata(10)
