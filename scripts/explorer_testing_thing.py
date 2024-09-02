# %%
from pathlib import Path
from saeco.evaluation.saved_acts_config import CachingConfig
from saeco.evaluation.chunk import Chunk
from saeco.evaluation.nnsite import getsite, setsite, tlsite_to_nnsite
from saeco.trainer import Trainable

from saeco.architectures.anth_update import cfg, anth_update_model

from jaxtyping import Int, Float
from torch import Tensor
from pydantic import BaseModel
from saeco.trainer.runner import TrainingRunner
import saeco.core as cl
import torch
from saeco.trainer.train_cache import TrainCache
from functools import wraps
from saeco.evaluation.evaluation_context import Evaluation
import nnsight

from rich.highlighter import Highlighter

# from transformers import GPT2LMHeadModel
ec = Evaluation.from_cache_name("ec_test")
# %%
nnsight_model = nnsight.LanguageModel("openai-community/gpt2", device_map="cuda")

# %%
import tqdm


import einops

ec.sae_cfg.train_cfg.data_cfg.model_cfg.acts_cfg.hook_site
ec.nnsight_model = nnsight_model

tl_name = "blocks.6.hook_resid_pre"
nn_name = tlsite_to_nnsite(tl_name)


# %%\
def active(document, position):
    return ec.saved_acts.acts[document][position]


active(4, 5)
from attr import define, field

from rich.console import Console

console = Console()
cursor_color = (255, 0, 0)
active_color = (0, 55, 255)
white = (255, 255, 255)


def color(c):
    return f"rgb({c[0]},{c[1]},{c[2]})"


@define
class Explorer:
    doc: int
    pos: int
    feat: int
    eval: Evaluation

    def print_activity(self):
        feature_activity = self.eval.saved_acts.acts[self.doc]
        document_id = self.doc
        tokens = self.document
        tokstrs = self.eval.detokenize(tokens)
        console.print(
            f"\n\n\n\nDocument {document_id}", style="underline bold", highlight=False
        )
        console.print("\n" + "-" * 30 + "\n", highlight=False)
        # if feature_activity.any():
        #     console.print(
        #         f"Feature {self.feat} active",
        #         [f"{i:.02}" for i in feature_activity.coalesce().values()],
        #         style=f"{color(active_color)} bold italic",
        #     )
        for i, t in enumerate(tokstrs):
            active: bool = False
            if i == self.pos:
                console.print(
                    "[",
                    style=f"{color(cursor_color)}  underline bold italic",
                    end="",
                    highlight=False,
                )
            if feature_activity[i, self.feat]:
                active = True
                console.print(
                    t,
                    style=f"{color(active_color)}  underline bold italic",
                    end="",
                    highlight=False,
                )
            if not active:
                console.print(t, style="rgb(255,255,255)", end="", highlight=False)
            if i == self.pos:
                console.print(
                    "]",
                    style=f"{color(cursor_color)}  underline bold italic",
                    end="",
                    highlight=False,
                )
        self.show_active_at_location()

    def active_at_location(self):
        return self.eval.saved_acts.acts[self.doc][self.pos].coalesce()

    def show_active_at_location(self):
        active = self.active_at_location()
        for i, v in enumerate(active.values()):
            if v:
                print(f"Feature {active.indices()[:,i]}: {v:.02}")

    @property
    def document(self):
        return self.eval.saved_acts.tokens[self.doc]

    def show_fwad(self):
        def make_tangent(acts):
            t = torch.zeros_like(acts)
            t[:, self.pos, self.feat] = 1
            return t

        def patch(acts):
            with torch.no_grad():
                z = torch.zeros_like(acts)
                z[:, self.pos, :] = acts[:, self.pos, :]
            return acts - z * 0

        out, tangent = self.eval.forward_ad_with_sae(
            tokens=self.document.unsqueeze(0),
            tangent_gen=make_tangent,
            patch_hook=patch,
        )
        tangent = (tangent + out.logits).softmax(-1) - out.logits.softmax(-1)
        tangent: Tensor = tangent.squeeze(0)
        self.view_top(tangent)

    def show_patch(self):

        def patch(acts):
            with torch.no_grad():
                acts[:, self.pos, self.feat] *= 0.99
            return acts

        diff = self.eval.patchdiff(self.document.unsqueeze(0), patch)
        self.view_top(-diff.squeeze(0))

    def view_top(self, tangent):
        top = tangent.topk(5, dim=-1)
        low = (-tangent).topk(5, dim=-1)
        doc = self.eval.detokenize(self.document)
        for i in range(len(doc)):
            if i == self.pos:
                print("\n<<<current position>>>\n")
            print(doc[i], end=" ")
            l = self.eval.detokenize(top.indices[i])
            for j in range(2):
                print(
                    f"{l[j]}: {top.values[i, j]:.02};",
                    end=" ",
                )
            l = self.eval.detokenize(low.indices[i])
            for j in range(2):
                print(
                    f"{l[j]}: -{low.values[i, j]:.02};",
                    end=" ",
                )
            print()


ex = Explorer(2, 15, 5, ec)
ex.pos = 17
ex.doc = 9959
ex.feat = 17
while True:
    ex.print_activity()
    ex.show_fwad()
    ex.show_patch()
    print()
# %%
