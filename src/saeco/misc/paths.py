import os
from pathlib import Path

SAVED_MODELS_DIR = os.environ.get(
    "SAECO_SAVED_MODELS_DIR", Path.home() / "workspace/saved_models/"
)
