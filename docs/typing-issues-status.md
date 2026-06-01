# Typing issues — status & tracker

Living tracker, reorganized **by status** (iterates on
[`typing-issues-to-review_reviewed.md`](typing-issues-to-review_reviewed.md),
which retains the full original per-finding dialogue and code context). The
active, decision-needing material is up top; everything **resolved** is
collapsed into the last section.

**Round-2 summary.** Working from your dispositions, this round committed:
`abecfad` (straightforward annotations), `1f6bfb5` (the review docs),
`549b50a` (merge of `dev`), and `62cb1ed` (all non-deferred resolutions below).
Gate after each: `ruff` clean on changed `src`; suites **library 222 /
sweepable 52 / research 16**.

Legend: `>` = your comment (round 1+2), `>>` = Claude reply, `>>>` = your reply.
Quotes are preserved verbatim where they carry the decision.

---

## 1. Deferred — needs your decision / a later pass

### High priority

- **`SweepData.save` / `sweep_id` int-vs-str divergence** —
  [`sweep_data.py:58`](../src/saeco/sweeps/sweep_data.py#L58). `sweep_id: str | None`
  but `save()` assigns `get_sweep_number()` (an `int`) on one branch and
  string-concatenates on the other; without `validate_assignment` the field can
  actually hold an int contradicting its annotation.
  > this is a good catch, and a high prio defer.

  >> Untouched. Fixing needs a chosen representation (likely `str`) and probably
  >> adjusting `get_sweep_number`'s return + its other caller — a real logic
  >> decision, not an annotation. Likely folds into the GenericBaseModel pass below.

- **✅ RESOLVED — `ArchRef` / `SweepData` → paramsight `GenericBaseModel` refactor** (tests `054ad87`, switch `0cddc87`). Covered
  [`from_json`](../src/saeco/architecture/arch_reload_info.py),
  `get_src`, `ArchClassRef`, and the runtime-specialization plumbing — all removed. (Details in §4. The `SweepData.sweep_id` int/str bug under §2 was **not** touched — still deferred.)
  > Yeah I think SweepData, ArchRef, etc could maybe be made into a GenericBaseModel
  > and then a lot of this logic could just be removed. When we do this, it should be
  > its own pass and we should add regression tests for all of these serialization
  > steps to make sure that it works the same before and after.
  > there may be other places which can similarly be simplified

  >> Deferred to its own pass. The `from_json`/`get_src` "safe sub-fixes" were NOT
  >> applied because you'd rather delete the logic via GenericBaseModel — re-surface
  >> when you start that pass and I'll write the serialization regression tests first.

  >>> paraphrased: greenlit as the next pass — you'll paint a specific picture of the
  >>> GenericBaseModel approach (and field questions) once I raise them. Note paramsight
  >>> does not export `GenericBaseModel` yet, so this is blocked on that feature + your
  >>> picture — see my questions in-channel.

### Medium priority

- **✅ RESOLVED this round — `TypeVarValue` switch for `get_config_class` / `get_arch_config_class`** (tests `fcb87f0`, switch `a005009`) —
  [`architecture.py:66`](../src/saeco/architecture/architecture.py#L66),
  [`sae_architecture.py:83`](../src/saeco/architecture/sae_architecture.py#L83).
  > yes you can go forward with the switch to TypeVarValue, do this in it's own pass/commit though.
  > [re the guard] the guard ought to be against both TypeVar and NoDefault ... We should address this in a later pass as the paramsight detail will get sorted out shortly.

  >> **Deferred this round, with reason** (the `get_args_at_base` rename WAS applied,
  >> commit `62cb1ed`). `paramsight.TypeVarValue` is a *descriptor*
  >> (`config_class = TypeVarValue[RunConfigT]()`), so it changes the call surface
  >> from a `.get_config_class()` method to an attribute (4 call sites:
  >> arch_reload_info.py:59, sweep_data.py:23/35/47). More importantly, because
  >> `RunConfigT` has a default (`= SweepableConfig`), the descriptor *resolves the
  >> base class to that default* rather than **raising** as the current guard does —
  >> which is precisely the "TypeVar **and** NoDefault" guard semantics you deferred.
  >> `TypeVarValueOption` (now present, `type[T] | None`) is the tool if you want
  >> "unresolved" to be `None` instead of raising. **One decision needed:** for the
  >> ungeneric base, do you want raise (`TypeVarValue` + drop the typevar default),
  >> return-the-default, or `None` (`TypeVarValueOption`)? Once you pick, this is a
  >> contained ~2-function + 4-caller pass. (paramsight is mid-change on the editable
  >> install — I'll verify against it when we do this.)
  
  >>>Oh to be clear, it will return the default if there is a default, not raise. If there is no default and there is no value
  >>>set for the typevar *then* it will raise.
  >>>So yeah just go to get_config_class->config_class and in the class definition we do config_class=TypeVarValue[RunConfigT]

  >>> AH I now understand what you're saying: That ArchitectureBase.config_class will fail to raise with this approach. I think
  >>>the fix is straightforward and also makes things more correct anyway: we can just remove the default. It seems to me like
  >>>it probably shouldn't be there and isn't expected to be there by anything else

  >> **Done** (`fcb87f0` regression tests, `a005009` switch): `config_class = TypeVarValue[RunConfigT]()`
  >> on `ArchitectureBase` and `arch_config_class = TypeVarValue[ArchConfigT]()` on `Architecture`;
  >> the `RunConfigT` default was removed so the bare base raises `LookupError` (preserving the guard).
  >> 4 call sites + the `define_architecture` example migrated; the disk_tensor `get_args_at_base`
  >> rename completed too. Regression tests (VanillaSAE/Gated resolved types + bare-base raise) green.
  
- **`SupportsResample.resample` Protocol contract** —
  [`features.py:17`](../src/saeco/components/features/features.py#L17). `FeaturesView.resample`
  handles `new_directions` as `Tensor` *or* `tuple[Tensor, Tensor]` while
  `SupportsResampleModule.resample` takes `Tensor`; the Protocol can't be typed
  until that disagreement is reconciled.
  > agreed on resolution, defer at the moment, please later resurface at which point I will decide the contract resolution/reconciliation.

  >> Re-surfacing here: when you're ready, decide whether `tuple` is part of the
  >> contract, then I'll type the Protocol + all implementers in one pass.

- **Normalizer scalar-identity convention** —
  [`normalizer.py:442`](../src/saeco/trainer/normalizers/normalizer.py#L442) (`mu_s`/`std_s`/`mu_e`/`std_e`)
  and [`normalizer.py:113`](../src/saeco/trainer/normalizers/normalizer.py#L113) (`AffineNormalizer.shift`).
  Several methods return a python scalar (`0`/`1`) in a "DONTUSE"/identity branch
  but are annotated `-> Tensor`.
  >> **No disposition was recorded for these in round 1** — left for you. It's a
  >> project-wide convention call (allow scalar broadcasting identities in the API,
  >> or force `torch.zeros_like`/`ones_like`). Tell me which and I'll apply it across
  >> the four methods + `shift` consistently.

- **`L0Targeter` PID knobs** —
  [`l0targeter.py:294`](../src/saeco/trainer/l0targeter.py#L294) (`__call__` returns
  `None` on unhandled `mode`, contradicting `-> float`) and
  [`l0targeter.py:231`](../src/saeco/trainer/l0targeter.py#L231) (`mode` etc. as bare
  str/bool on an unstable research surface).
  >> No disposition recorded; the doc frames it as "with the author." The `__call__`
  >> None-leak is a real latent bug (a final `raise`/exhaustive handling fixes it);
  >> the `mode` Literal/StrEnum is a research-knob design call. Flag if you want
  >> either now.

### Low priority

- **`NormFeatures` / `OrthogonalizeFeatureGrads` Protocol-vs-nominal bound** —
  [`feature_hooks.py:58`](../src/saeco/components/hooks/feature_hooks.py#L58).
  > yeah this is a great catch and a defer. Ideal would be to intersect Module with Protocol. But no intersection type. Ty has an intersection type, so if at some point I migrate to ty ... we will just use ty's intersection type. Otherwise we will switch to representing it with an ABC ... but yeah for now deferring on this

- **`ActsScaledByDecoderNormPenaltyMixin.decoder`** — what it stores is
  inconsistent with how `penalty()` reads it
  ([`penalty_mixins.py:5`](../src/saeco/components/penalties/penalty_mixins.py#L5)).
  > yeah good catch, under-exercised code path. Defer for now

  >> (Its `penalty` *return* was typed `-> Tensor` this round; the `self.decoder`
  >> reconciliation is untouched.)

- **`to_losses` nnsight-proxy return** —
  [`recons.py:182`](../src/saeco/trainer/recons.py#L182). `>defer low prio`
- **`Trainer.subject_model`** (nnsight surface) — `>defer`. **`save_callback`**
  Protocol ([`trainer.py:52`](../src/saeco/trainer/trainer.py#L52)) — `>defer this, currently its broken iirc anyways.`
- **`mlog.create_sweep`** interface reconciliation
  ([`mlog_.py:43`](../src/saeco/mlog/mlog_.py#L43)) — `>defer`. **`CustomSweeper`/`BaseLogger`**
  keep-or-remove triage ([`fns.py:30`](../src/saeco/mlog/fns.py#L30)) — `>defer`.

---

## 2. New issues discovered this round

Found while implementing; **not** resolved (each needs a small decision or its
own change):

- **`EMAFreqTracker.freqs` — assert-not-none is unsafe (re-raising to you).**
  [`ema.py:17`](../src/saeco/components/resampling/freq_tracker/ema.py#L17).
  > does assert not none work here? if so, do that, if not re-raise this issue to me.

  >> Re-raised: **no.** In `src/` `.freqs` is always populated before read (assert
  >> would be safe), but `research/` has a genuine *access-before-init* path, so
  >> `assert self.activation_freqs is not None` would break research. Options: (a)
  >> seed `activation_freqs` to a tensor at construction (runtime/behaviour change);
  >> (b) widen the property to `Tensor | None` (forces None-guards at resampler call
  >> sites + diverges from the base's `-> Tensor`); (c) fail-fast in the research
  >> path. Your call — left untouched.

- **Duplicate pydantic field declarations** —
  [`data_cfg.py:78-83`](../src/saeco/data/config/data_cfg.py#L78). `databuffer_queue_size`,
  `databuffer_worker_queue_base_size`, `databuffer_worker_offset_mult` are each
  declared twice. Latent (last decl wins); worth de-duplicating in a small cleanup.

- **`Schedule` / `RunSchedule` / `ResampleSchedule` may also be dead** —
  [`scheduler.py`](../src/saeco/trainer/scheduler.py). The `Scheduler` stub was
  deleted this round; these three have no importers anywhere either. Candidate
  for module removal after confirming no dynamic/string-based use.

- **`BaseLogger.config_get` stub inconsistency** —
  [`fns.py:453`](../src/saeco/mlog/fns.py#L453). `def config_get(self): ...` returns
  `None`, unlike the two sibling classes whose `config_get` returns `self`. Likely
  a stub bug; rolls into the deferred `CustomSweeper`/`BaseLogger` triage.

- **`matry_clt` research recons no longer conforms** —
  `research/.../matry_clt/matryoshka_clt.py:96`. `get_multisite_recons_loss` still
  uses the old `cast_fn=...` / `data_cfg=None` shape and is declared against
  `ReconstructionEvaluatorFunctionProtocol`, which changed this round. Research
  scratch; update when that arch is next touched.

---

## 2b. Inbox follow-ups (from `typing-issues-todos_misc_inbox.md`)

New requests dropped mid-round — tracked here, not yet implemented.

- **`Loss.loss(x, y, y_pred, cache)` — type `x`/`y`/`y_pred`.**
  >>> paraphrased: can we get types on x, y, y_pred on the loss methods?
  >> Doable and small: in the L2 / Cosine implementations these are `torch.Tensor`
  >> (input acts, target, reconstruction). Will confirm against the `Loss` base + all
  >> overrides/callers (incl. research) before pinning.

- **`TruncatedLoss`: mixin → wrapper.**
  >>> paraphrased: do `TruncatedLoss` as a wrapper that wraps a loss rather than a
  >>> mixin — likely "less nasty".
  >> A design refactor (behavioral structure, not just typing). Worth doing; I'd scope
  >> the wrapper shape and confirm before changing behavior — flag if you want it now.

- **`Lambda.__init__` — `module` / `func` types.**
  >>> paraphrased: `module` looks like it could be `Module | None` (verify?); `func`
  >>> looks like `Callable[[Tensor | DictBatch], Tensor | DictBatch]`.
  >> **Verified — `module` is NOT `Module | None`:** `Lambda.forward` explicitly
  >> dispatches on `nn.Parameter` (a Tensor subclass, *not* an `nn.Module` — e.g.
  >> `co.Lambda(lambda x: x.detach(), self.mag)`) plus a final arbitrary-callable
  >> branch, so the honest type is `nn.Module | nn.Parameter | Callable[..., Any] | None`.
  >> `func` ≈ your proposal, but note it is also called with `module(x)`'s output, not
  >> only the forward input; `Callable[[Tensor | DictBatch], Tensor | DictBatch]` is a
  >> reasonable (slightly loose) choice.

## 3. Ignore / leave-as-is (deliberate, no action)

- **Arch-prop methods** (`VanillaSAE.model`/`L2_loss`/… + gated) — inference
  already types them. `> these seem fine left unannotated since inference is able to handle it.`
- **`AnthResamplerConfig.enc_directions` / `dec_directions`** — int defaults are
  by-design pydantic coercion. `> lgtm`
- **`Cache.__call__`'s `-> "SubCacher | T"`** — deliberate IDE-autocomplete lie.
  `> agreed, leave as is.`
- **`Module.forward`** — introspected `cache:` annotation + subclass-dependent
  return. `>good`
- **`ModelReloadInfo.source_backup`** — pydantic field on a `@deprecated` model.
  `> ignore deprecated case ->skip`
- **`geo_med.getmed`/`getmean` `buf`/`normalizer` params** — no live callers
  (only commented-out usage). `> resolution seems good.`

---

## 4. Resolved this round  (commits `62cb1ed`, `fcb87f0`, `a005009`, `054ad87`, `0cddc87`)

### `ArchRef` / `SweepData` → `GenericBaseModel` (`054ad87` tests, `0cddc87`)
`ArchRef[ConfigT, ArchT]` and `SweepData[ConfigT, ArchT]` are now
`paramsight.GenericBaseModel` subclasses, so both type args (the run-config type
and the Architecture class) round-trip through JSON automatically. `ArchClassRef`,
`get_src`, and `ArchRef.from_json` are removed; the arch class is carried as the
`ArchT` generic param, recovered via the `cfg_t`/`arch_t` `TypeVarValue` descriptors;
deserialize is `model_validate_json` (GBM reconstructs the specialization). `xcls`
override kept explicit in `load_arch`; `sweep_data`/`sweep_runner` call sites
rewritten to the new accessors. The nested-generic arg (`RunConfig[VanillaConfig]`)
round-trips correctly. Regression tests pin config-type preservation + `.load`.

### Architecture config-class → `TypeVarValue` descriptors (`fcb87f0` tests, `a005009`)
`get_config_class()` / `get_arch_config_class()` methods replaced by descriptors
`ArchitectureBase.config_class = TypeVarValue[RunConfigT]()` and
`Architecture.arch_config_class = TypeVarValue[ArchConfigT]()`; `RunConfigT` default
removed so the bare base raises `LookupError`. Callers (arch_reload_info, sweep_data)
+ the define_architecture example migrated; `get_resolved_typevars_for_base` →
`get_args_at_base` completed in disk_tensor{,_collection}. Regression tests pin the
resolved types and the bare-base raise.


Condensed; full original rationale per item is in
[`typing-issues-to-review_reviewed.md`](typing-issues-to-review_reviewed.md).
All verified against callers (incl. `research/`); ruff-clean; suites green.

### Config domains → `StrEnum` (values == prior strings; pydantic still coerces persisted configs)
| where | change |
|---|---|
| [`optim_reset.py`](../src/saeco/components/features/optim_reset.py#L134) | `b2_technique` → `B2Technique`; `b2_techniques` dict → exhaustive `match` handler fn |
| [`resampler.py`](../src/saeco/components/resampling/resampler.py#L62) | `freq_balance_strat` → `FreqBalanceStrat`; comparisons updated |
| [`train_config.py`](../src/saeco/trainer/train_config.py#L58) | `l0_targeter_type` → `L0TargeterType`; `optim` → `OptimType` (incl. `ScheduleFree`) |
| [`optim_config.py`](../src/saeco/trainer/optim_config.py#L5) | registry hoisted to module level; `get_optim_cls -> type[Optimizer] \| Callable[..., Optimizer]` |

### Penalty family / losses
- `Penalty.penalty`, `LambdaPenalty.penalty`, `SummedPenalties.penalty`, both
  penalty mixins → `-> Tensor`; int-`0` seeds replaced with device/dtype-correct
  scalar zero tensors; `LinearDecayPenaltyMixin`'s early `torch.zeros(1)` device
  bug fixed.
- [`losses.py`](../src/saeco/components/losses/losses.py#L39) `SparsityPenaltyLoss.loss`
  → `-> torch.Tensor` + `assert isinstance(l, torch.Tensor)`.

### Cache / Trainer state
- `sparsity_penalty: torch.Tensor` and `forward_reuse_dict` (keys are `(int,int)`,
  not `str` — typed accordingly) mirrored across
  [`sae_cache.py`](../src/saeco/components/sae_cache.py#L7),
  [`reused_forward.py`](../src/saeco/core/reused_forward.py#L10),
  [`train_cache.py`](../src/saeco/trainer/train_cache.py); ellipsis sentinel kept.
- `TrainCache.trainer: "Trainer" = ...` watched field added; `make_cache`
  placeholder line dropped.
- Thread A safe parts: `handle_backward`/`handle_step`/`trainstep`/`eval_step`/
  `Trainer.log` `-> None`, `_train` `num_steps: int | None`, `handle_backward`
  `loss: torch.Tensor` (param narrowings deferred).

### `Trainable` cluster ([`trainable.py`](../src/saeco/trainer/trainable.py))
`loss` `assert isinstance(...) + -> Tensor`; `forward` `x: torch.Tensor | SAETrainBatch`;
`rearrange -> tuple[Tensor, torch.Size | None]` + `dearrange` unpacks `doc, seq, _`;
`get_acts` `cache: Cache | None` + `@overload`s for `pre_acts`; `param_groups -> list[dict[str, Any]]`.

### recons → conforms to `ReconstructionEvaluatorFunctionProtocol`
[`recons.py`](../src/saeco/trainer/recons.py#L53) + [`evaluation_protocol.py`](../src/saeco/trainer/evaluation_protocol.py#L9):
`cast_fn` default `...` → `contextlib.nullcontext`; `data_cfg` required;
`-> dict[str, float]`; `tokens`/`num_batches`/`batch_size` typed.

### Behaviour fixes (verified callers discard / unaffected)
- [`resampler.py`](../src/saeco/components/resampling/resampler.py#L125) `assign_model`
  now `return self` (was returning the arg; latent bug — all 15 call sites discard it).
- [`paths.py`](../src/saeco/misc/paths.py#L4) `SAVED_MODELS_DIR` coerced to `Path`
  (broke when the env var was set).
- [`generation_config.py`](../src/saeco/data/config/generation_config.py#L21)
  `num_act_piles`: `assert acts_per_pile is not None` guard + `int -> int`.

### Deletions / dead-code
- `tensorclass` ([disk_tensor.py](../src/saeco/data/storage/disk_tensor.py)) and the
  `Scheduler` stub ([scheduler.py](../src/saeco/trainer/scheduler.py)) deleted.
- [`piler.py`](../src/saeco/data/piler/piler.py#L196) `__getitem__`: contract pinned
  to `int | slice | list[int]` (verified callers incl. downstream `comlm`); dead
  else-branch + `# type: ignore` removed; `dict_piler.py` mirrored (unused `cast` dropped).
- [`data_cfg.py`](../src/saeco/data/config/data_cfg.py#L130) unused
  `target_gb_per_pile` param removed (no caller passed it).

### Local annotations / misc
- `get_args_at_base` rename (architecture.py, sae_architecture.py).
- [`optim_reset.py`](../src/saeco/components/features/optim_reset.py#L18)
  `OptimFieldResetValue.get_value -> Tensor | float`, `feat_mask: Tensor` + bool
  assert (the `ft_optim_field_state`/`new_directions` params were left untyped to
  stay consistent with the `OptimFieldFeatures` caller).
- [`anthropic_resampling.py`](../src/saeco/components/resampling/anthropic_resampling.py#L19)
  `ModuleIO.inputs: list[tuple[Tensor, ...]]`; **4 new tests** in
  [`test_module_io.py`](../tests/components/resampling/test_module_io.py).
- [`tools.py`](../src/saeco/initializer/tools.py#L35) `layer`/`mlp_layer` typed
  (`nonlinearity: type[nn.Module] | nn.Module`); `reused` full PEP 695 ParamSpec.
- [`fnlambda.py`](../src/saeco/components/ops/fnlambda.py#L7) `func: Callable`;
  [`kernels_fns.py`](../src/saeco/components/jumprelu/kernels_fns.py#L77) `centrality(kernel: Kernel, ...)`.
- [`propagator.py`](../src/saeco/core/collections/propagator.py#L40) private
  reduction attrs → `… | None`.
- [`disk_tensor_collection.py`](../src/saeco/data/storage/disk_tensor_collection.py#L43)
  `_remove_finalized_disktensor -> None`.
- [`exception_location_hint.py`](../src/saeco/misc/exception_location_hint.py#L4)
  `cache: Cache` via TYPE_CHECKING import.
- [`locations.py`](../src/saeco/data/config/locations.py#L22) dir props → `cached_property`.
- [`tokens_data.py`](../src/saeco/data/training_data/tokens_data.py#L252)
  `iter_docs_and_columns` params (`batch_size: int`, `columns: list[str] | None`).
- [`mlog/fns.py`](../src/saeco/mlog/fns.py#L316) `_sweep_inst_config: dict | None`
  (now a real attrs field, fixing a latent slots AttributeError).
