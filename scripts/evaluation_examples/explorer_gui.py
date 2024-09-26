# %%
from functools import wraps
from pathlib import Path

import circuitsvis

import circuitsvis.tokens
import nnsight
import saeco.core as cl
import torch

from jaxtyping import Float, Int
from load import ec
from nicegui import ui
from pydantic import BaseModel

from rich.highlighter import Highlighter

from saeco.analysis.uiitem import UIE
from saeco.architectures.anth_update import anth_update_model, cfg
from saeco.evaluation.evaluation import Evaluation
from saeco.evaluation.nnsite import getsite, setsite, tlsite_to_nnsite
from saeco.evaluation.saved_acts_config import CachingConfig
from saeco.evaluation.storage.chunk import Chunk
from saeco.trainer import Trainable
from saeco.trainer.runner import TrainingRunner
from saeco.trainer.train_cache import TrainCache
from torch import Tensor

# from transformers import GPT2LMHeadModel
# %%
nnsight_model = nnsight.LanguageModel("openai-community/gpt2", device_map="cuda")

import einops

# %%

# %%
import tqdm

from metadata_test import filt_eval

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

inactive = white
active_0 = torch.tensor([100, 0, 180])
active_1 = torch.tensor([255, 0, 0])


def colorstr(c):
    return f"rgb({c[0]},{c[1]},{c[2]})"


def colorprint(s, c, end, style=""):
    console.print(s, style=f"{colorstr(c)} {style}", end=end, highlight=False)


@define(slots=False)
class Explorer:
    doc: int
    pos: int
    feat: int
    eval: Evaluation
    filt_eval: Evaluation | None

    def __attrs_post_init__(self):
        self.current_document
        self.button
        if self.filt_eval is None:
            self.filt_eval = self.eval

    def update(self): ...
    @UIE
    def button(self, cb):
        return ui.button("click me", on_click=cb)

    @UIE
    def current_document(self, cb):
        return ui.card()

    @current_document.updater
    def current_document(self, e: ui.card):
        e.clear()
        feature_activity = self.eval.saved_acts.acts[self.doc].to_dense()[:, self.feat]
        tokens = self.document
        tokstrs = self.eval.detokenize(tokens)

        def set_pos(i):
            def set():
                self.pos = i
                self.update()

            return set

        max_activity = feature_activity.max()

        content = list(enumerate(zip(tokstrs, feature_activity)))
        style_normal = f"border: 1px solid #f0f0f0; border-radius: 8px; padding: 1px; margin: 1px; margin-top: -10px"
        # e.style("gap: 0.1rem")
        with e:
            for j in range(0, len(content), 8):
                rowcontent = content[j : j + 8]
                with ui.row() as row:
                    row.style(
                        "margin: 1px 0; padding: 0px; border: 0px; border-radius: 0px; gap: 0.1rem"
                    )
                    for i, (tok, feat) in rowcontent:
                        if " " in tok:
                            ui.label(" ")
                            ui.label(" ")
                            ui.label(" ")
                            ui.label(" ")
                        color = (
                            active_0 + (active_1 - active_0) * feat / max_activity
                        ).long()
                        if feat == 0:
                            color = inactive
                        btn = ui.label(tok)
                        btn.on("click", set_pos(i))
                        if i == self.pos:
                            btn.style(
                                f"margin: 0px; padding: 0px; border: 4px solid #2222aa; border-radius: 2px; background-color: {colorstr(color)};"
                            )
                        else:
                            btn.style(
                                f"margin: 0px; padding: 0px; border: 0px; border-radius: 0px; background-color: {colorstr(color)};"
                            )
        # e.content = h.local_src
        # ui.add_body_html(h.local_src)
        # self.show_active_at_location()

    def active_at_location(self):
        return self.eval.saved_acts.acts[self.doc][self.pos].coalesce()

    def show_active_at_location(self):
        active = self.active_at_location()
        min = active.values().min()
        max = active.values().max()
        cmin = torch.tensor([255, 0, 0])
        cmax = torch.tensor([0, 255, 0])
        print()
        colorprint(f"min: {min}", cmin, " ")
        colorprint(f"max: {max}", cmax, " \n")
        for i, v in enumerate(active.values()):
            fn = active.indices()[:, i]
            colorprint(
                fn,
                (cmin + (cmax - cmin) * (v - min) / (max - min)).long(),
                ", ",
                # "bold",
            )

    @property
    def document(self):
        return self.eval.saved_acts.tokens[self.doc]

    def show_fwad(
        self,
        est_prob_deltas=False,
        feature_shrink=0,
        shrink_wrt_pos=True,
        shrink_wrt_feat=True,
    ):
        def make_tangent(acts):
            t = torch.zeros_like(acts)
            t[:, self.pos, self.feat] = 1
            return t

        def patch(acts):
            if feature_shrink == 0:
                return acts
            with torch.no_grad():
                z = torch.zeros_like(acts)
                if shrink_wrt_pos and shrink_wrt_feat:
                    z[:, self.pos, self.feat] = acts[:, self.pos, self.feat]
                elif shrink_wrt_pos:
                    z[:, self.pos, :] = acts[:, self.pos, :]
                elif shrink_wrt_feat:
                    z[:, :, self.feat] = acts[:, :, self.feat]
                else:
                    # if feature_shrink == 0:
                    return acts * (1 - z)

                    # raise ValueError("No shrinking specified")
            return acts - z * feature_shrink

        out, tangent = self.eval.forward_ad_with_sae(
            tokens=self.document.unsqueeze(0),
            tangent_gen=make_tangent,
            patch_fn=patch,
        )
        if est_prob_deltas:
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

    def tokens_to_feats(self):
        to_feat_attrib = self.eval.forward_token_attribution_to_features(
            self.document, seq_range=(0, self.pos)
        )
        tokstrs = self.eval.detokenize(self.document)
        for i in range(self.pos):
            attrib = to_feat_attrib[i, self.pos, self.feat]
            if torch.any(attrib != 0):

                print(tokstrs[i], attrib)
            else:
                print(tokstrs[i])

    def nt(self):
        self.pos += 1

    def pt(self):
        self.pos -= 1

    def nf(self):
        active = self.active_at_location()
        idxs = active.indices()[0]
        mask = idxs == self.feat
        if not mask.any():
            self.feat = idxs[0]
            return
        cf = mask.nonzero().squeeze().item() + 1
        if cf == len(idxs):
            self.feat = idxs[0]
            return
        self.feat = idxs[cf]
        self.print_activity()

    def pf(self):
        active = self.active_at_location()
        idxs = active.indices()[0]
        values = active.values()
        mask = idxs == self.feat
        if not mask.any():
            self.feat = idxs[0]
            return
        cf = mask.nonzero().squeeze().item() - 1
        if cf == len(idxs):
            self.feat = idxs[0]
            return
        self.feat = idxs[cf]
        self.print_activity()


exp = Explorer(62, 15, 5527, ec, filt_eval)
ui.run()

# ex = Explorer(2, 15, 5, ec)
# ex.pos = 17
# ex.doc = 62
# ex.feat = 5527
# ex.tokens_to_feats()

# ex.nf()
# ex.pf()
# while True:
#     ex.print_activity()
#     ex.show_fwad(feature_shrink=0.5, est_prob_deltas=True)
#     ex.show_patch()
#     print()
# # %%
