from saeco.components.sae_cache import SAECache


class TrainCache(SAECache):
    L2_loss = ...
    sparsity_penalty = ...
    L2_aux_loss = ...
    loss = ...
    L1 = ...
    L0 = ...
    L1_full = ...
    L0_aux = ...
    cosim = ...
    trainstep = ...
