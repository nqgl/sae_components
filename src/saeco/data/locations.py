from pathlib import Path


class DataDirLocations:
    def __init__(self) -> None:
        self.initialized = False

    def initialize(self):
        if self.initialized:
            return
        self.initialized = True
        self.top_level_dir = Path.home()

    def customize(self, **kwargs):
        if self.initialized:
            raise ValueError("Cannot customize after initialization")
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def dir_prop(m):
        def wrapper(self, *args, **kwargs) -> Path:
            if not self.initialized:
                self.initialize()
            d = m(self, *args, **kwargs)
            if not d.exists():
                d.mkdir()
            return d

        return property(wrapper)

    @dir_prop
    def TOP_DIR(self) -> Path:
        return Path.home()

    @dir_prop
    def SAVE_DIR(self) -> Path:
        SAVE = self.top_level_dir / "workspace"
        if not SAVE.exists():
            SAVE.mkdir()
        return SAVE

    @dir_prop
    def DATA_DIR(self) -> Path:
        return self.SAVE_DIR / "data"

    @dir_prop
    def _CHUNKS_DIR(self) -> Path:
        return self.DATA_DIR / "chunks"

    @dir_prop
    def TOK_CHUNKS_DIR(self) -> Path:
        return self._CHUNKS_DIR / "tokens"

    @dir_prop
    def ACT_CHUNKS_DIR(self) -> Path:
        return self._CHUNKS_DIR / "acts"

    @dir_prop
    def CACHE_DIR(self) -> Path:
        return self.SAVE_DIR / "cache"


DATA_DIRS = DataDirLocations()
