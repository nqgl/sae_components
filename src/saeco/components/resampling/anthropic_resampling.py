from typing import ClassVar

import torch
import torch.nn as nn
from attrs import define

from saeco.components.resampling.resampler import Resampler, ResamplerConfig
from saeco.core import Cache

# from collections import defaultdict


@define
class ModuleIO:
    inputs: dict[nn.Module, list[torch.Tensor]]
    outputs: dict[nn.Module, list[torch.Tensor]]
    model_out: torch.Tensor


def get_modules_io(model: nn.Module, modules: list[nn.Module], data):
    inputs = {m: [] for m in modules}
    outputs = {m: [] for m in modules}

    def hook(module, args, output):
        inputs[module].append(args)
        outputs[module].append(output)

    handles = []
    for module in modules:
        handles.append(module.register_forward_hook(hook))

    out = model(data.float())
    for handle in handles:
        handle.remove()
    return ModuleIO(
        inputs=inputs,
        outputs=outputs,
        model_out=out,
    )


def get_param_parent_module(param, model: nn.Module):
    modules = model.named_modules()
    containing = {}
    for name, module in modules:
        for mparam in module.parameters(recurse=False):
            if param is mparam:
                containing[name] = module
    assert len(containing) == 1
    return list(containing.values())[0]


from enum import IntEnum

from saeco.sweeps import SweepableConfig


class ResampleType(IntEnum):
    enc_in = 0
    error = 1
    model_in = 2


from saeco.components.features.optim_reset import (
    OptimResetValues,
    OptimResetValuesConfig,
)


class AnthResamplerConfig(ResamplerConfig):
    enc_directions: ResampleType = 0
    dec_directions: ResampleType = 1
    ResampleType: ClassVar = ResampleType


class AnthResampler(Resampler):
    cfg: AnthResamplerConfig

    @torch.no_grad()
    def get_reset_feature_directions(self, num_directions, data_source, model):
        # gotta make sure to treat the normalization correctly!
        # and also the centering bits
        # ... hmm
        # a cute/elegant thing would be to look for the
        # first resampled module and get the input at that point
        # and then use that to get the directions
        errors = []
        errmags = []
        enc_inputs = []
        enc_module = self.get_encoder_containing_module(
            model.model
        )  # TODO hmm what's going on with this
        inputs = []
        need_enc_in = self.cfg.enc_directions == 0 or self.cfg.dec_directions == 0
        for i in range(10):
            data = next(data_source).float()
            res = get_modules_io(model, modules=[enc_module], data=data.input)
            enc_inputs_list = res.inputs[enc_module]
            if len(enc_inputs_list) != 1:
                for i in range(len(enc_inputs_list) - 1):
                    if torch.any(enc_inputs_list[i][0] != enc_inputs_list[i + 1][0]):
                        raise ValueError("Expected one input")
            enc_input = enc_inputs_list[0][0] if need_enc_in else None
            error = data.target - res.model_out
            assert error.ndim == 2
            errmag = error.norm(dim=1)
            inputs.append(data.input)
            errors.append(error)
            errmags.append(errmag)
            enc_inputs.append(enc_input)
        errmag = torch.cat(errmags)
        if errmag.shape[0] <= errmag.shape[0]:
            i = torch.arange(num_directions)
        else:
            v, i = torch.topk(errmag, k=errmag.shape[0])

        error = model.normalizer(torch.cat(errors)[i], cache=Cache())
        inputs = model.normalizer(torch.cat(inputs)[i], cache=Cache())
        enc_inputs = torch.cat(enc_inputs)[i] if need_enc_in else None

        # mag = 0.01
        direction_types = [enc_inputs, error, inputs]
        return (
            direction_types[self.cfg.enc_directions],
            direction_types[self.cfg.dec_directions],
        )

    def get_encoder_containing_module(self, model: nn.Module):
        assert len(self.encs) == 1
        enc = self.encs[0]

        return get_param_parent_module(enc.param, model)

    def get_decoder_containing_module(self, model: nn.Module):
        assert len(self.decs) == 1
        dec = self.decs[0]
        return get_param_parent_module(dec.param, model)

        # could instead find the common ancestor or something
        # -> store the module tree by name in nested dicts
        # or have a get modules that contain module


class AResampler(Resampler):
    @torch.no_grad()
    def get_reset_feature_directions(self, num_directions, data_source, model):
        # gotta make sure to treat the normalization correctly!
        # and also the centering bits
        # ... hmm
        # a cute/elegant thing would be to look for the
        # first resampled module and get the input at that point
        # and then use that to get the directions
        errors = []
        errmags = []
        enc_inputs = []
        enc_module = self.get_encoder_containing_module(model.model)
        inputs = []
        for i in range(10):
            data = next(data_source)
            res = get_modules_io(model, modules=[enc_module], data=data)
            enc_inputs_list = res.inputs[enc_module]
            assert len(enc_inputs_list) == 1
            enc_input = enc_inputs_list[0][0]
            error = data - res.model_out
            assert error.ndim == 2
            errmag = error.norm(dim=1)
            inputs.append(data)
            errors.append(error)
            errmags.append(errmag)
            enc_inputs.append(enc_input)
        errmag = torch.cat(errmags)
        v, i = torch.topk(errmag, k=errmag.shape[0] // 2 + 1)
        error = model.normalizer(torch.cat(errors)[i], cache=Cache())
        inputs = model.normalizer(torch.cat(inputs)[i], cache=Cache())
        enc_inputs = torch.cat(enc_inputs)[i]
        adj = (error[1:] * error[:-1]).sum(dim=1)
        vadj, iadj = torch.topk(adj, k=num_directions)
        errora = error[iadj]
        errorb = error[iadj + 1]
        errora /= errora.norm(dim=1, keepdim=True)
        errorb /= errorb.norm(dim=1, keepdim=True)
        # shared =

        # mag = 0.01
        direciton_types = [enc_inputs, error, inputs]
        return (
            direciton_types[self.cfg.enc_directions],
            direciton_types[self.cfg.dec_directions],
        )

    def get_encoder_containing_module(self, model: nn.Module):
        assert len(self.encs) == 1
        enc = self.encs[0]

        return get_param_parent_module(enc.param, model)

    def get_decoder_containing_module(self, model: nn.Module):
        assert len(self.decs) == 1
        dec = self.decs[0]
        return get_param_parent_module(dec.param, model)
