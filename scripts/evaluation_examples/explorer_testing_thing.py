# %%

import nnsight
import torch
from load import root_eval
from torch import Tensor

from saeco.evaluation.evaluation import Evaluation
from saeco.misc.nnsite import tlsite_to_nnsite

# from transformers import GPT2LMHeadModel
# %%
nnsight_model = nnsight.LanguageModel("openai-community/gpt2", device_map="cuda")


# %%

# %%

root_eval.nnsight_model = nnsight_model

tl_name = "blocks.6.hook_resid_pre"
nn_name = tlsite_to_nnsite(tl_name)


# %%\
def active(document, position):
    return root_eval.saved_acts.acts[document][position]


active(4, 5)
from attr import define
from rich.console import Console

console = Console()
cursor_color = (255, 0, 0)
active_color = (0, 55, 255)
white = (255, 255, 255)


def color(c):
    return f"rgb({c[0]},{c[1]},{c[2]})"


def colorprint(s, c, end, style=""):
    console.print(s, style=f"{color(c)} {style}", end=end, highlight=False)


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


ex = Explorer(2, 15, 5, root_eval)
ex.pos = 17
ex.doc = 62
ex.feat = 5527
ex.tokens_to_feats()

ex.nf()
ex.pf()
while True:
    ex.print_activity()
    ex.show_fwad(feature_shrink=0.5, est_prob_deltas=True)
    ex.show_patch()
    print()
# %%
