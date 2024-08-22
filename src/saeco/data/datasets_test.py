import datasets
from pydantic import BaseModel

from saeco.data.sc.SplitConfig import SplitConfig

datasets.ReadInstruction(
    "train",
).to_spec()


class C(BaseModel):
    r: str | None = datasets.ReadInstruction("train", from_=10, unit="%").to_spec()


c = C()

print(c.model_dump())


from saeco.data import DataConfig, ModelConfig, ActsDataConfig

c = DataConfig(
    dataset="cyrilzhang/TinyStories-tokenized-gpt2-1024",
    trainsplit=SplitConfig(
        splitname="train", tokens_from_split=1_000_000, start=0, end=20
    ),
    seq_len=128,
    model_cfg=ModelConfig(),
)
next(c.get_databuffer())
# transformers.token
# from transformers import AutoModel, AutoTokenizer
# import os

# AutoTokenizer.from_pretrained(
#     "google/gemma-2b", token=os.environ["HUGGINGFACE_API_KEY"]
# )
# AutoModel.from_pretrained("google/gemma-2b", token=os.environ["HUGGINGFACE_API_KEY"])
