import saeco.core as cl


class IndexerGenerator:
    def __getitem__(self, item) -> "Indexer":
        return Indexer(item)


class Indexer(cl.Module):
    def __init__(self, slicer):
        super().__init__()
        self.slicer = slicer

    def forward(self, x, *, cache: cl.Cache, **kwargs):
        return x[self.slicer]

    L = IndexerGenerator()
