from api import create_app
from saeco.evaluation import Evaluation

STORAGE_NAME = "stored_acts"
root = Evaluation.from_cache_name(STORAGE_NAME)
app = create_app(root)
