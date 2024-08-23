from pydantic import BaseModel


class DataGenerationProcessConfig(BaseModel):
    """
    Nothing in here matters wrt training behavior.
    This just changes the data generation process in ways that effect efficiency and file size.
    It should be fine to not touch these parameters, unless you're running out of memory, in which case you should try reducing the batch sizes.
    """

    tokens_per_pile: int = 2**27  # ~100m tokens * 8 bytes = ~1gb/pile
    acts_per_pile: int | None = 2**18
    # TODO decide if we're using this or using gb per pile
    # 2048 dd & fp16 has 2^20 acts at 1m * 2k * 2 = 4gb
    # so 2 ** 18 gives 1gb piles
    target_gb_per_pile: float | None = 1
    meta_batch_size: int = 2**19  # how many tokens to send in each meta-batch
    llm_batch_size: int = 2**17  # batch size when running the LLM
    num_document_distribution_batches: int = 100

    def num_act_piles(self, num_tokens):
        if self.acts_per_pile is None:
            bytes_per_pile = self.target_gb_per_pile * 2**30
            dtype_bytes = 2  # hardcoded assumption of float16
            b_per_act = self.model_cfg.acts_cfg.d_data * dtype_bytes
            total_b = num_tokens * b_per_act
            return 1 + (total_b + bytes_per_pile - 1) // bytes_per_pile
        return 1 + num_tokens // self.acts_per_pile
