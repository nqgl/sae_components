import datasets
from pydantic import BaseModel

from saeco.data.config.split_config import SplitConfig

datasets.ReadInstruction(
    "train",
).to_spec()


class C(BaseModel):
    r: str | None = datasets.ReadInstruction("train", from_=10, unit="%").to_spec()


c = C()

print(c.model_dump())


from saeco.data import (
    DataConfig,
    ModelConfig,
)

c = DataConfig(
    dataset="cyrilzhang/TinyStories-tokenized-gpt2-1024",
    trainsplit=SplitConfig(splitname="train", start=4, end=5),
    seq_len=128,
    model_cfg=ModelConfig(model_name="gemma-2b"),
    # generation_config=DataGenerationProcessConfig(llm_batch_size=2**15),
)
i = 0
model = c.model_cfg.model
for d in c._get_databuffer():
    i += 1

print(i)
print(c.get_split_tokens("train", num_tokens=1_000_000).shape)
# print(c.get_split_tokens("train", num_tokens=100).shape)
# transformers.token
# from transformers import AutoModel, AutoTokenizer
# import os

# AutoTokenizer.from_pretrained(
#     "google/gemma-2b", token=os.environ["HUGGINGFACE_API_KEY"]
# )
# AutoModel.from_pretrained("google/gemma-2b", token=os.environ["HUGGINGFACE_API_KEY"])
