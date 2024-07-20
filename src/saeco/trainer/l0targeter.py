import torch
from typing import Protocol, Optional
from .schedule_cfg import RunSchedulingConfig


class L0TargeterProto(Protocol):
    target: float
    schedule: RunSchedulingConfig

    def __call__(self, l0: float, t: int) -> float: ...


class MultiEma:
    def __init__(self, betas, weights=None, inv_betas=True, reweight=True):
        if isinstance(betas, float):
            betas = torch.tensor([betas]).cuda()
        elif isinstance(betas, list):
            betas = torch.tensor(betas).cuda()
        if isinstance(weights, list):
            weights = torch.tensor(weights).cuda()
        if inv_betas:
            betas = 1 - betas
        self.velocities = betas == 1
        if reweight:
            rescale = torch.where(~self.velocities, 1 / (1 - betas), 1)
        else:
            rescale = torch.ones_like(betas)
        if weights is not None:
            assert weights.shape == betas.shape
            weights = weights * rescale
        else:
            weights = rescale
        assert ((0 <= betas) & (betas <= 1)).all()

        self.weights = weights

        self.betas_inv = 1 - betas
        self.ema = torch.zeros(self.betas_inv.shape[0]).cuda()

    @torch.no_grad()
    def update(self, x, beta_mul=1):
        self.ema.lerp_(x, self.betas_inv * beta_mul)
        self.ema[self.velocities] += x

    @property
    def value(self):
        return self[:]
        return (self.ema * self.weights).sum() / self.weights.sum()

    def __getitem__(self, i):
        return (self.ema[i] * self.weights[i]).sum() / self.weights[i].abs().sum()

    def coherency(self, i):
        ema_values = self.ema[i] * self.weights[i] / self.weights[i].abs().sum()
        v = ema_values.sum()
        ss = torch.sum(ema_values.sign())
        if ss.sign() != v.sign():
            return torch.zeros(1).cuda()
        return v * ss.abs() / len(ema_values)

    def adjachency(self, i):
        ema_values = self.ema[i] * self.weights[i] / self.weights[i].abs().sum()
        v = ema_values.sum()
        signs = ema_values.sign()
        sames = signs[:-1] == signs[1:]
        sw = torch.where(sames, ema_values[:-1] + ema_values[1:], 0).sum()
        ss = torch.sum(ema_values.sign())
        if ss.sign() == sw.sign() == v.sign():
            return sw.sum()
        return torch.zeros(1).cuda()

    def decay(self, beta):
        self.update(torch.zeros(1).cuda(), beta)


def diffema(a, q, n=0.1):
    assert 0 < q < 1 and 0 < a < 1 and 0 < n < 0.5, (a, q, n)
    b = q * a
    c = (1 - (1 - n) * q) * a / n
    assert 0 < b < 1 and 0 < c < 1, (a, b, c)
    assert c > a > b, (a, b, c)
    return (
        [a, b, c],
        [1, n - 1, -n],
    )


def lfema(a, n=0.1):
    c = a / n
    assert 0 < c < 1
    return MultiEma(
        [a, c],
        weights=[1, -n],
        reweight=False,
    )


def lfema2(a, c1, c2, x):
    w1 = (a**x - a * c2 ** (x - 1)) / (c1**x - c1 * c2 ** (x - 1))
    w2 = (a - w1 * c1) / c2
    assert -1 < w1 < 1 and -1 < w2 < 1, (w1, w2)
    assert w1 + w2 < 1

    return MultiEma(
        [a, c1, c2],
        weights=[1, -w1, -w2],
        reweight=False,
    )


def diffemas(al, ql, nl):
    maxlen = max(len(al), len(ql), len(nl))
    al += [al[-1]] * (maxlen - len(al))
    ql += [ql[-1]] * (maxlen - len(ql))
    nl += [nl[-1]] * (maxlen - len(nl))

    bl = []
    wl = []

    # for a, q, n in zip(al, ql, nl):
    for a in al:
        for q in ql:
            for n in nl:
                b, w = diffema(a, q, n)
                bl += b
                wl += w
    bd = {}
    for i, w in enumerate(wl):
        bd.setdefault(bl[i], []).append(w)
    bl2 = []
    wl2 = []
    for k, l in bd.items():
        bl2.append(k)
        wl2.append(sum(l))
    return MultiEma(
        bl2,
        weights=wl2,
        reweight=False,
    )


class CMultiEma(MultiEma):
    def __getitem__(self, i):
        return self.coherency(i)


class AMultiEma(MultiEma):
    def __getitem__(self, i):
        return self.adjachency(i)


class L0Targeter(L0TargeterProto):
    def __init__(
        self,
        l0_target: Optional[float],
        schedule: RunSchedulingConfig,
    ):

        self.target = torch.tensor([l0_target]).cuda()
        self.schedule = schedule

        self.inv = False

        self.i = MultiEma([0.001, 0.0003], weights=[1, 2], reweight=False)
        self.p = lfema(0.002)
        # MultiEma([0.01], reweight=False)
        self.velocity = 0

        self.d = diffemas([0.01], [0.3, 0.5], [0.1])
        # diffemas([0.011, 0.003, 0.05, 0.1], [0.4, 0.5], [0.07, 0.1])
        # MultiEma(
        #     [0.08, 0.03, 0.2],
        #     weights=[1, -0.9, -0.1],
        #     reweight=False,
        # )
        self.dd = MultiEma([0.01, 0.007])
        # MultiEma(
        #     [0.007, 0.01, 0.003, 0.0012],
        #     # [0.003 * 1.62 ** (-i) for i in range(8)],
        #     # weights=[1, 2, 1],
        #     reweight=False,
        # )

        self.a = diffemas([0.03, 0.05], [0.1], [0.1])
        #
        # diffemas([0.05, 0.1], [0.3, 0.5], [0.2, 0.1])
        # MultiEma(
        #     [0.4, 0.2],
        #     weights=[1, -1],
        #     reweight=False,
        # )
        self.aa = MultiEma(
            [0.01],
            # [0.01 * (1.6 ** (-i)) for i in range(4)],
            reweight=False,
        )
        self.dd2 = MultiEma(
            [0.001],
            # [0.01 * (1.6 ** (-i)) for i in range(4)],
            reweight=False,
        )
        self.dd2.ema += 1

        self.mo = torch.zeros(1).cuda()
        self.mo_beta = 0.1

        self.p_c = 1
        self.i_c = 0.5
        self.d_c = 10
        self.a_c = 0
        self.scale = 0.25
        self.velocity_mode = False
        self.direct_mode = True
        self.by_sign = False
        self.mode = "clamp_all"
        self.last_dsign = 0
        self.last_flip_t = 0
        self.last_flip_interval = 100

    @property
    def I(self):
        return self.i.value * self.i_c

    @property
    def D(self):
        return self.d.value * self.d_c / self.dd2.value**2

    @property
    def A(self):
        return self.aa.value * self.a_c

    @property
    def P(self):
        return self.p.value * self.p_c

    def update(self, l0, t):
        pos = (torch.log(l0) - torch.log(self.target)) if self.inv else l0 - self.target

        self.p.update(pos)
        self.i.update(self.p.value)
        # self.d.decay(0.01)
        self.d.update(self.p.value)
        # self.dd.decay(0.999)
        self.dd.update(self.d.value)
        # self.a.decay(0.001)
        self.a.update(self.d.value)
        self.aa.update(self.a.value)
        dsign = self.dd.value.sign()
        if dsign != self.last_dsign:
            interval = t - self.last_flip_t
            if interval < 1000:
                if self.last_flip_interval < 1000:
                    self.dd2.ema += 1.5
                self.dd2.ema += 0.3
            self.last_flip_interval = interval
            self.last_flip_t = t
        else:
            self.dd2.update(torch.ones(1).cuda() * 0.3)
        self.last_dsign = dsign

    def __call__(self, l0: float, t: int) -> float:
        self.update(l0, t)
        modir = self.P + self.I + self.D + self.A
        if self.by_sign:
            modir = modir.sign() + self.P.sign() + self.I.sign() + self.D.sign()
        self.last_modir = modir
        if self.P.sign() == (s := self.D.sign()) and s != modir.sign():
            modir = modir * 0
            # return 0
        if self.velocity_mode:
            self.mo += modir * 0.003
            step = (self.mo) + self.D * 0.2
        elif self.direct_mode:
            step = modir
        else:
            self.mo.lerp_(modir, self.mo_beta)
            step = self.mo
        step *= self.scale
        # if self.P.sign() == (s := self.D.sign()) and s != step.sign():
        #     self.mo *= 0.9
        #     return 0
        if self.mode == "step":
            return step.item()
        if self.mode == "clamp":
            return step.clamp(-1, 1).item()
        if self.mode == "clamp_all":
            return (
                step.clamp(-1, 1)
                + self.P.clamp(-1, 1)
                + self.I.clamp(-1, 1)
                + self.D.clamp(-1, 1)
                + self.A.clamp(-1, 1)
            ).item()

    def loggables(self, t):
        if t < 100 or t % 3 != 0:
            return {}

        return {
            # "targeting_PID/ema": self.ema.value,
            "targeting_PID/mo": self.mo,
            "targeting_PID/P": self.P,
            "targeting_PID/d": self.d.value,
            "targeting_PID/dd": self.dd.value,
            "targeting_PID/aad": self.dd2.value,
            "targeting_PID/a": self.a.value,
            "targeting_PID/aa": self.aa.value,
            "targeting_PID/I": self.I,
            "targeting_PID/D": self.D,
            "targeting_PID/A": self.A,
            "targeting_PID/last_modir": self.last_modir,
        }


# class L0Targeter(L0TargeterProto):
#     def __init__(
#         self,
#         l0_target: Optional[float],
#         schedule: RunSchedulingConfig,
#     ):
#         self.target = l0_target
#         self.schedule = schedule

#     def __call__(self, l0: float, t: int) -> float:
#         # if not self.schedule.dynamic_adjust(t):
#         #     return 0
#         gentle_zone_radius = 1
#         distance = abs(l0 - self.target) / gentle_zone_radius
#         stepscale = min(
#             1,
#             (distance * 6 + 1) / 7,
#         )

#         return (-1 if self.target > l0 else 1) * stepscale

#     def loggables(self, t):
#         return {}
