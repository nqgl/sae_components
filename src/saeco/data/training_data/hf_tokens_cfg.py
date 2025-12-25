
from saeco.sweeps import SweepableConfig


class HFTokensConfig(SweepableConfig):
    dataset: str = "alancooney/sae-monology-pile-uncopyrighted-tokenizer-gpt2"
    tokens_column_name: str = "input_ids"
    seq_len: int | None = 128
    set_bos: bool = True
    perm_all: bool = False
