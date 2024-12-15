from saeco.sweeps import SweepableConfig


class DataGenerationProcessConfig(SweepableConfig):
    """
    Nothing in here matters wrt training behavior.
    This just changes the data generation process in ways that effect efficiency and file size.
    It should be fine to not touch these parameters, unless you're running out of memory, in which case you should try reducing the batch sizes.
    """

    tokens_per_pile: int = 2**26  # ~100m tokens * 8 bytes = ~1gb/pile
    acts_per_pile: int | None = 2**15  # Num activations per pile
    # eg with d_dict 2048 & fp16 2^20 acts at 1m * 2k * 2 = 4gb so 2^18 = ~1gb/pile
    meta_batch_size: int = 2**16  # how many tokens to send in each meta-batch
    llm_batch_size: int = 2**12  # batch size when running the LLM (in tokens)
    num_document_distribution_batches: int = 100

    def num_act_piles(self, num_tokens):
        return 1 + num_tokens // self.acts_per_pile
