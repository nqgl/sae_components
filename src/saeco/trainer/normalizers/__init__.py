from .normalizer import Normalizer, L2Normalizer, ConstL2Normalizer, Normalized

NORMALIZERS_LIST = [L2Normalizer, ConstL2Normalizer]
NORMALIZERS = {n.__name__: n for n in NORMALIZERS_LIST}
