import saeco.analysis.wandb_analyze as wandb_analyze

baseline_runs_name = "L0Targeting_cmp/f5jbxrmd"
targeted_runs_name = "L0Targeting_cmp/zvvtvwt0"


def get_close_pairs(l1: list[int], l2: list[int]) -> list[tuple[int, int]]:
    """
    returns the pairs of indices in l1 and l2 that are closest
    """
    l1se = sorted(list(enumerate(l1)), key=lambda x: x[1])
    l2se = sorted(list(enumerate(l2)), key=lambda x: x[1])
    return [(l1se[i][0], l2se[i][0]) for i in range(len(l1))]
