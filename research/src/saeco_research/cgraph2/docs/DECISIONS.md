# cgraph2 — design decisions log

Append-only. Each entry: the call, the options, the choice, the reasoning, and
the alternative we didn't take. Significant calls only (model/API/semantics).

---

## D-01 — GraphSpec is the runnable context (fold in ValueGraph)

**Call:** Should the runnable + registration layer be separate (`ValueGraph`) from
the authoring layer (`GraphSpec`)?

**Choice:** No — make a `GraphSpec` instance itself runnable. It registers its own
producers (so it's a real `cl.Module`: `.parameters()`, `.to()`, `state_dict`),
and exposes `.forward_values` / `.collect` / `forward`. `ValueGraph` is retired
into it.

**Why:** Glen's "a ref only means something in relation to its graph." A handle is
a node, but to *run* it you need its producers registered on a module, and that
context is the graph. Bundling handles + registration + run into one object makes
"access the handle through its graph" literal and kills the `ValueGraph(output,
(extras...))` hand-assembly smell. Matches "everything is a graph."

**Mechanism (as built):** `GraphSpec` stays a *plain object* (not an nn.Module).
It runs off the *live* module references its nodes already hold (no registration
needed to compute), so it gets `forward_values`/`collect`/`forward` directly. For
the nn.Module bookkeeping a trainable needs (`.parameters()`/`.to()`/state_dict),
`g.register_into(module)` walks the cone and registers the leaf producers onto the
owning module. `GraphSpec.of(**handles)` builds one dynamically for runtime-sized
aggregates (the trainable's whole computation). `ValueGraph` retired.

**Why not nn.Module:** the original plan (make `_GraphMeta` ABCMeta-based,
`GraphSpec(cl.Module)`) collides head-on with `g(x)` = *apply* (D-02 composition):
`nn.Module.__call__` is *run*. Keeping `g(x)` = apply (so `model(x).acts`
composes) and separating "run" (`g.forward_values`) from "register"
(`g.register_into`) is cleaner and avoids the conflict. The trainable registers
by calling `self.graph.register_into(self)`.

---

## D-02 — Keep nested-subgraph structure (don't flatten); dedup is the execution view

**Call:** When graph g1 composes into g2, is g1 flattened away, or kept?

**Choice:** Kept. g2 keeps its bound sub-graphs as navigable children (`g2.g1`,
`g2.g1.n`), preserving the boundary. The globally-interned node DAG is the
*execution* view (interning gives the dedup/fusion); the nested children are the
*structural* view. They coexist and are consistent (children's handles point into
the interned DAG).

**Why:** Glen: "keep the whole subgraph in the representation, really... not just
leave a trace." Inlining-vs-bubble-up was a false tension — keep structure for
navigation/bubble-up, derive the deduped DAG for running.

**Note on mechanism:** Glen described computing the deduped path at *compute* time
and caching it. We get the same deduped DAG eagerly via interning at construction
(equivalent result, already "cached" by construction). Recorded as a possible
future change if eager interning ever gets in the way; not needed now.

---

## D-03 — Bubble-up metrics (not model-exposed handles)

**Call:** Metrics (L0/L1) as (a) bubbled-up sites collected from the graph, or
(b) plain handles the model graph exposes by name.

**Choice:** (a), bubble-up — Glen: "I do not like B, I like bubble up metrics
being available." A site (e.g. `ActMetrics`) lives on `acts` wherever the model
puts it; the trainable `collect`s it from the graph cone and derives its handles
from the collected node's (already-rebound) input, so the metric is computed in
the full normalized context for free.

**Why:** information bubbles up from where it lives without the outside wiring it —
the saeco ActMetrics property, made graph-native. Reserves the machinery for the
stateful case too (freq tracker's deferred update).

**Alternative:** (b) was simpler but Glen explicitly wants bubble-up.

---

## D-04 — Total loss is a fused graph handle (weights as scalar source-nodes)

> **SUPERSEDED (overnight self-review).** The weight *buffers* are unsafe across
> devices: `nn.Module._apply` (`.to(device)`) **reassigns** a buffer to a new
> tensor, so the `ref(w, ())` source-node (captured at init) and `self._weights[k]`
> orphan on CPU while the registered buffer moves to CUDA — verified on GPU
> (registered buffer on cuda, node's tensor on cpu, a dead duplicate in
> state_dict; it only "worked" via silent `cuda * cpu-0dim` promotion). Reverted to
> **device-safe Python weighting** (`total = sum(term * coeff)`): the loss terms
> still share one fused `forward_values` pass (the fusion that matters), and the
> dynamic coeff weight is applied in Python. A genuine `total_loss` *handle* needs
> weights as real graph **inputs** — awaits multi-input/`DictBatch`. Lesson: a
> graph node captures a *tensor*; only `nn.Parameter` survives `.to()` in place, so
> graph-referenced state must be a param or a module that reads its own live buffer,
> never a bare registered buffer.


**Call:** Is the weighted total loss a Python sum of saved terms, or a handle the
graph computes?

**Choice:** A handle. Each loss term is multiplied by a per-loss scalar weight
*source node* (`ref(weight_buffer, ())`), summed into `total_loss`. `loss()`
syncs the weight buffers from `coeffs` (so L0-targeting's dynamic sparsity weight
flows in), runs one `forward_values` pass, and `save(total_loss)` is the backprop
loss; the *unweighted* terms are also saved for logging (matching saeco's
`cache.<lossname>` = unweighted, `cache.loss` = weighted sum).

**Why:** Glen: "we have some input that's the weights on the losses, then we
multiply each by its corresponding loss and we sum" — the total loss *is* a graph.
Execution-equivalent to Python weighting (the terms share one fused pass either
way), but `total_loss` becomes a first-class handle, which is the graph-native form.

**Note:** weights are scalar *buffers* synced each step — a pragmatic stand-in for
true graph *inputs*. The cleaner form passes the weight vector as an input node,
which awaits multi-input/`DictBatch`. Recorded for later.

---

## D-05 — ActMetrics is a GraphSpec; a thin site marker makes it bubble-up-able

**Call:** "Make ActMetrics a GraphSpec" (Glen) vs. it must be `collect`-able as a
site. Tension: a `GraphSpec` *inlines* when composed (no marker node to collect),
and a `GraphSpec` instance can't be a passthrough producer (its `__call__` is
apply, not identity).

**Choice:** Split the two roles. `ActMetrics(GraphSpec)` is the reusable metric
*computation* (`acts -> L0/L1`, passthrough `output`). A thin `_MetricsSite`
producer (real passthrough) is what gets spliced onto `acts` (`metered(acts)`) and
`collect`ed; it carries the metrics graph class and, on collection, applies it to
the *rebound* acts (`site.at(node.inputs[0])`) — so the metric is computed in the
full normalized context (the site node sits on the recon path, so it rebinds with
the model). Metrics thus bubble up (D-03): the model buries a site on its acts, the
trainable `collect`s it, and nothing in between declares L0/L1.

**Why:** keeps the metric a clean composable graph *and* collectable *and*
full-context. The site marker is plumbing; `ActMetrics` is the graph.

**Alternative not taken:** make the GraphSpec instance itself the producer
(breaks: `__call__` = apply) or tag the passthrough with `self` in the class body
(breaks: no instance at class-def time).
