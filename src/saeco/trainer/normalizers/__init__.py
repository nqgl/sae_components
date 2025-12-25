from .normalizer import (
    Aggregation,
    ConstL2Normalizer,
    GeneralizedNormalizer,
    GNConfig,
    L2Normalizer,
    Normalized,
    Normalizer,
)

NORMALIZERS_LIST = [L2Normalizer, ConstL2Normalizer]
NORMALIZERS = {n.__name__: n for n in NORMALIZERS_LIST}
