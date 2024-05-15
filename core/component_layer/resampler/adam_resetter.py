import torch


class AdamResetter:
    def __init__(self, _object, sq_ema_reset_ratio=1):
        self._obj_to_reset_fields_of = _object
        self._sq_ema_reset_ratio = sq_ema_reset_ratio

    def __getattribute__(self, __name: str) -> torch.Any:
        if __name in ["_obj_to_reset_fields_of", "_sq_ema_reset_ratio"]:
            return super().__getattribute__(__name)
        return AdamParamResetter(
            getattr(self._obj_to_reset_fields_of, __name), self._sq_ema_reset_ratio
        )


class AdamParamResetter:
    def __init__(self, param, sq_ema_reset_ratio=1, transpose=None):
        self.param = param
        self._sq_ema_reset_ratio = sq_ema_reset_ratio
        self._transpose = transpose

    def transpose(self, *args, **kwargs):
        return AdamParamResetter(
            self.param, self._sq_ema_reset_ratio, transpose=(args, kwargs)
        )

    def __getitem__(self, indices):
        return AdamResetterCallable(
            self.param,
            indices,
            sq_ema_reset_ratio=self._sq_ema_reset_ratio,
            transpose=self._transpose,
        )


class AdamResetterCallable:
    def __init__(self, param, indices, sq_ema_reset_ratio=None, transpose=None):
        self.param = param
        self.indices = indices
        self._sq_ema_reset_ratio = sq_ema_reset_ratio or 1
        self._transpose = transpose

    @torch.inference_mode()
    def __call__(
        self,
        adam: torch.optim.RAdam,
        alive_indices=None,
        sq_ema_reset_ratio=None,
        reset_momentum=True,
    ):
        state = adam.state[self.param]
        exp_avg = (
            state["exp_avg"]
            if self._transpose is None
            else state["exp_avg"].transpose(*self._transpose[0], **self._transpose[1])
        )
        exp_sq = (
            state["exp_avg_sq"]
            if self._transpose is None
            else state["exp_avg_sq"].transpose(
                *self._transpose[0], **self._transpose[1]
            )
        )
        if reset_momentum:
            exp_avg[self.indices] = 0  # zero momentum
        eps = 1e-7
        ratio = 1
        sq_ema_reset_ratio = sq_ema_reset_ratio or self._sq_ema_reset_ratio
        if alive_indices is None:
            assert False, "alive_indices == None is not implemented"
            state["exp_avg_sq"][self.indices] = (
                eps
                + torch.sum(state["exp_avg_sq"])
                - torch.sum(state["exp_avg_sq"][self.indices] * ratio)
            ) / (eps + state["exp_avg_sq"].numel() - self.ind_numel() * ratio)
        else:
            alive = exp_sq[alive_indices]
            exp_sq[self.indices] = (
                sq_ema_reset_ratio * (eps + torch.sum(alive)) / (eps + alive.numel())
            )

    def ind_numel(self):
        print("ind_numel called not implemented")
        return 5
