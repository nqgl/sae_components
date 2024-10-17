import fastapi
import uvicorn
from saeco.evaluation import Evaluation
from saeco.evaluation.api import create_app

STORAGE_NAME = "mid_store"
root = Evaluation.from_cache_name(STORAGE_NAME)
app = create_app(fastapi.FastAPI(), root)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
