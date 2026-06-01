from functools import cached_property, wraps
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
        @wraps(m)
        def wrapper(self, *args, **kwargs) -> Path:
            if not self.initialized:
                self.initialize()
            d = m(self, *args, **kwargs)
            if not d.exists():
                d.mkdir()
            return d

        return wrapper

    @cached_property
    @dir_prop
    def TOP_DIR(self) -> Path:  # noqa: N802  # constant-like path accessor; SCREAMING_CASE intentional
        return Path.home()

    @cached_property
    @dir_prop
    def SAVE_DIR(self) -> Path:  # noqa: N802  # constant-like path accessor; SCREAMING_CASE intentional
        save = self.top_level_dir / "workspace"
        if not save.exists():
            save.mkdir()
        return save

    @cached_property
    @dir_prop
    def DATA_DIR(self) -> Path:  # noqa: N802  # constant-like path accessor; SCREAMING_CASE intentional
        return self.SAVE_DIR / "data"

    @cached_property
    @dir_prop
    def _CHUNKS_DIR(self) -> Path:  # noqa: N802  # constant-like path accessor; SCREAMING_CASE intentional
        return self.DATA_DIR / "chunks"

    @cached_property
    @dir_prop
    def TOK_CHUNKS_DIR(self) -> Path:  # noqa: N802  # constant-like path accessor; SCREAMING_CASE intentional
        return self._CHUNKS_DIR / "tokens"

    @cached_property
    @dir_prop
    def ACT_CHUNKS_DIR(self) -> Path:  # noqa: N802  # constant-like path accessor; SCREAMING_CASE intentional
        return self._CHUNKS_DIR / "acts"

    @cached_property
    @dir_prop
    def CACHE_DIR(self) -> Path:  # noqa: N802  # constant-like path accessor; SCREAMING_CASE intentional
        return self.SAVE_DIR / "cache"


DATA_DIRS = DataDirLocations()
