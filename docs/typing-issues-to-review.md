# Typing issues left for human judgement

This document catalogues **72 typing findings that were deliberately NOT
auto-fixed**. A separate pass already applied ~77 straightforward annotation
fixes (adding obvious return types, tightening `dict` -> `dict[str, X]` where
unambiguous, etc.). What remains here are the cases where a "correct"
annotation is a *judgement call*, a *cross-file contract change*, or
*entangled with runtime semantics* — the things a script should not decide on
its own. These are surfaced for the maintainer to weigh, not bugs to fix
blindly.

## Why these are thorny — the recurring root causes

Most findings fall into one of a few families. Read these once; the per-finding
sections lean on them.

### 1. Runtime-semantic annotations

In several subsystems the annotation is not just documentation — it *is* the
runtime behaviour. Changing it changes what the program does:

- **`SweepableConfig` / pydantic fields.** The field's annotation `T` drives
  pydantic validation **and** is widened at runtime to
  `T | Swept[T] | SweepExpression[T]`. Tightening e.g. `str -> Literal[...]`
  genuinely narrows what configs validate, and can reject persisted/old sweep
  configs that load today. These are config-schema decisions, not type tweaks.
  *(Landmine.)*
- **attrs `@define` fields.** Inside an `@define` class, an *annotated*
  assignment is collected as a real field (with an init param / slot); an
  *unannotated* one is a plain class attribute. Adding an annotation can change
  construction semantics. Field-type changes should be reviewed as such.
- **`Cache` watched-attribute sentinel (`= ...`).** Cache subclass fields use
  `= ...` as a "declared but unset" sentinel managed by the Cache machinery
  (not pydantic). The annotation is runtime-semantic-adjacent, and the same
  field is sometimes declared in several files with divergent types.
- **Introspected method annotations.** `cache:` parameter annotations on
  `Module.forward`-family methods are read by
  `saeco.components.type_acc_methods` to derive cache dispatch; arch-descriptor
  return types (`@model_prop`/`@loss_prop`/...) are the generic the descriptor
  keys off of. Changing these shifts inferred types at every call site.

### 2. Dynamic surfaces

Some values come from libraries whose static types are intentionally loose or a
lie at rest:

- **nnsight** is `exec`/dynamic; saved proxies (`.save()`) behave like Tensors
  only after the trace context exits, so annotating them `Tensor` is a
  convenient fiction about the proxy lifecycle.
- **torch optimizers / state dicts / param groups** are heterogeneous
  `dict[str, Any]`-shaped bags consumed by dynamic constructors; the honest
  type is only marginally better than `dict`.
- **HF dataset row-slices** index dynamically and yield `Any`.

### 3. Dynamic generic specialization (paramsight)

`paramsight` does runtime generic specialization — `cls[config_class]` produces
a runtime-specialized `BaseModel` subclass, and `get_resolved_typevars_for_base`
is typed `type | GenericAlias | None`. The precise generic parameter is only
known at runtime, so the static return type is genuinely unexpressible; the
already-annotated siblings carry a `reportReturnType` complaint today. Adding
matching annotations improves intent but propagates that same checker
disagreement.

**None of these are bugs to fix blindly.** Where a finding *does* contain a safe
sub-fix (e.g. "the return is clean, only the param is thorny"), that is called
out explicitly.

---

## architecture

All three findings here are instances of **dynamic generic specialization
(paramsight)** plus the **safe-part/thorny-part split**: each has a clean
sub-annotation and a genuinely unexpressible one.

### `src/saeco/architecture/arch_reload_info.py:53` — `ArchRef.from_json`
- Current: `def from_json(cls, d: dict, xcls: "type[ArchitectureBase[Any]] | None" = None):`
- Issue: two issues bundled. `d` is JSON-loaded and string-indexed (`d["class_ref"]`), and there is no return annotation.
- Why thorny: the return is `archref_cls.model_validate(d)` where `archref_cls = cls[config_class]` is a runtime-specialized BaseModel subclass — unexpressible to a checker (pyright already flags `cls[config_class]` at line 60 as "already specialized"). `ArchRef[Any]` / `Self` are both approximations.
- Resolution: tighten only `d: dict[str, Any]` (clean, safe). Leave the return for a human to weigh against the existing checker noise.

### `src/saeco/architecture/sae_architecture.py:83` — `Architecture.get_arch_config_class`
- Current: `def get_arch_config_class(cls):`
- Issue: returns the resolved arch-config type from `get_resolved_typevars_for_base(cls, Architecture)[0]`, guarded against being a TypeVar, so intent is `-> type[ArchConfigT]` (matching sibling `ArchitectureBase.get_config_class -> type[RunConfigT]`). `@takes_alias` does not introspect the return, so this is pure-typing at runtime.
- Why thorny: `get_resolved_typevars_for_base` is typed `type | GenericAlias | None`, so `-> type[ArchConfigT]` does **not** type-check cleanly — the sibling carries this exact annotation and pyright reports `reportReturnType` on it today (architecture.py:76). Adding it here propagates the same disagreement.
- Resolution: (a) add the annotation for intent/consistency, accepting the already-precedented checker noise; or (b) add a localized cast/`# type: ignore`. Human call.

### `src/saeco/architecture/arch_reload_info.py:15` — `get_src`
- Current: `def get_src(obj):`
- Issue: both branches return `str`; `obj` must expose `__module__` and be inspectable.
- Why thorny: paramsight's `ObjRef.from_obj` (the in-repo caller) types the analogous arg `FunctionType | type`, but `get_src` is also reachable from `get_obj` on arbitrary loaded objects, so a too-narrow param could over-constrain a caller passing another callable kind — verifying the full caller set crosses into the paramsight package.
- Resolution: add `-> str` (safe). Leave the param for a human who can confirm callers in paramsight.

---

## arch-build

A mixed bag, but two threads dominate: **introspected arch-descriptor return
types** and **research-only helpers with no library callers** (low priority,
outside the API contract).

### `src/saeco/architectures/vanilla/vanilla_model.py:45` — `VanillaSAE.model / decoder / L2_loss / sparsity_loss` (and the analogous methods in `gated/arch.py`)
- Current: arch-prop-decorated methods (`@model_prop def model(self): return SAE(...)`) with no return annotation.
- Issue: intent is clear from each body (`model_prop -> SAE`, `loss_prop -> a Loss`, etc.).
- Why thorny (**introspected annotation**): the `arch_prop` decorators are PEP 695 generics (`model_prop[SAE_T: nn.Module]`, ...) whose bound type parameter is inferred from the decorated method's return type and propagated through `__get__` overloads to every `self.model`/`self.<loss>` access. pyright **already infers these correctly from the bodies**, so adding `-> SAE` is redundant-but-explicit, not a fix. Because the annotation is the thing the descriptor generic keys off of, changing it shifts inferred types at every call site. (`FieldRegistry` discovers by descriptor *type*, not by reading the annotation, so discovery is unaffected.)
- Resolution: (a) annotate all arch-props explicitly for documentation/consistency; or (b) leave inference to do its job. Consistency-vs-redundancy design call.

### `src/saeco/initializer/tools.py:33` — `layer / mlp_layer` (nonlinearity param + return)
- Current: `def layer(d_in, d_out, nonlinearity=nn.LeakyReLU, scale=1.0):`
- Issue: both functions return `Seq`; dims/scale are clearly `int`/`float`.
- Why thorny: `nonlinearity` is dual-use — a class (checked `isinstance(..., type)`, instantiated, special-cased for `nn.PReLU`) **or** an already-constructed instance. The precise type `type[nn.Module] | nn.Module` is awkward and gives checkers little leverage (both classes and instances are callable). Research-only code, no library callers.
- Resolution: add `-> Seq` and the int/float params if pursued; the `nonlinearity` union is a deliberate choice. Low priority.

### `src/saeco/initializer/tools.py:21` — `reused`
- Current: memoizing decorator factory `def reused(func): ...`.
- Issue: a faithful PEP 695 ParamSpec signature is expressible: `def reused[**P, T: Module](func: Callable[P, T]) -> Callable[P, ReuseForward[T]]`.
- Why thorny: memoization means the returned callable *ignores* its args after the first call, so a ParamSpec signature overstates how args are used; and `ReuseForward` is generic over a `Module` bound. Non-trivial generic-decorator annotation for a research-only helper with no library callers.
- Resolution: invest in the full generic signature only if someone wants the inference; otherwise leave.

### `src/saeco/initializer/geo_med.py:21` — `getmed/getmean` (buf, normalizer params)
- Current: `def getmed(buf, normalizer, num_batches=10):`
- Issue: `buf` is `Iterator[Tensor]` (`next(buf)`), `normalizer` is a `Normalizer` (`normalizer.normalize(x, cache=Cache())[0]`).
- Why thorny: **no live callers** (the only call in `runner.py:55` is commented out). The `Normalizer` base does not expose a simple typed `normalize` (it routes through `cl.Module`), and `buf`'s concrete protocol is not pinned here. Pinning `normalizer: Normalizer` would add a cross-package import from trainer into the initializer subsystem for a dead path.
- Resolution: leave until a caller is reinstated and the buffer protocol is settled.

---

## components-features

### `src/saeco/components/features/features.py:17` — `SupportsResample.resample`
- Current: fully untyped `def resample(self, *, indices, new_directions, bias_reset_value, optim): ...`.
- Issue: this `runtime_checkable` Protocol's method is untyped while its concrete implementers are typed.
- Why thorny (**cross-file Protocol reconciliation**): the Protocol must stay compatible with **both** `FeaturesView.resample` (features_param.py:193, `new_directions` handled as `Tensor` *or* `tuple[Tensor, Tensor]`) and `resampler.SupportsResampleModule.resample` (`new_directions: Tensor`). The two **disagree** on `new_directions`, so picking one signature can make the other stop satisfying the protocol under a strict checker. Tightening also ripples to every implementer (resampler.py, research/`other_lin.py` overrides).
- Resolution: reconcile the `resample` contract across implementers first (decide whether `tuple` is part of the contract), then type the Protocol. Not a local edit.

### `src/saeco/components/features/optim_reset.py:18` — `OptimFieldResetValue.get_value`
- Current: abstract method, `...` body, no return annotation (so checker infers `-> None`; every concrete override is flagged incompatible, e.g. `ResetToConst` returns `self.num`).
- Issue: real contract is "a value usable on the RHS of an optim-state assignment": `ResetToConst -> int/float`, the `ResetTo*`/`*SqMean*` variants `-> Tensor`. Honest base is `Tensor | float`.
- Why thorny: abstract contract with several in-file overrides; the exact union is a judgement call (`Tensor | float | int` vs `Tensor | float` — `int` is subsumed by `float`). The three params (`ft_optim_field_state`, `feat_mask` a bool index Tensor, `new_directions` a Tensor) should be settled together so the signature is coherent, since this sets the override-compatibility bar for all subclasses.
- Resolution: confirm the union and type the whole signature in one pass.

### `src/saeco/components/features/optim_reset.py:134` — `OptimResetValuesConfig.b2_technique`
- Current: `b2_technique: str = "sq"`.
- Issue: only ever used as a key into `b2_techniques` (line 154), whose keys are exactly `{"mean","sq","axsq","axmean"}`. Real domain is a 4-value Literal; an out-of-domain value is a silent `KeyError` at resample time instead of a validation failure.
- Why thorny (**SweepableConfig landmine**): annotation drives pydantic validation and is Swept-widened. `str -> Literal[...]` narrows what validates and could reject currently-loading configs — but genuinely improves fail-fast.
- Resolution: decide deliberately as a config-schema change; if adopted, keep the literal members in sync with `b2_techniques.keys()`.

---

## components-hooks-penalties

The dominant root cause here is **the `Penalty` family return contract**. Six
findings are all "the `penalty(self, x, *, cache)` method returns a `Tensor` in
every real use, but annotating the base/abstract/mixin return imposes a contract
on the whole hierarchy (including research/ scratch)." They should be decided as
one coherent decision to type the `Penalty` family, not piecemeal. The shared
caveats:

- The abstract base (`Penalty.penalty`, penalty.py:11) and `LambdaPenalty`
  (penalty.py:11/16/20) have **user-supplied callables** whose return the class
  cannot prove is a `Tensor`; `-> Tensor` documents an *asserted*, not enforced,
  contract.
- The two **standalone mixins** (`penalty_mixins.py:12`, `:32`) call
  `super().penalty(...)` with no base of their own, so `super()` is unresolvable
  in isolation — a checker cannot verify the delegated branch without the
  concrete MRO.
- `SummedPenalties.penalty` (l1_penalizer.py:90) seeds an `int 0` accumulator,
  so a checker sees `int | Tensor` (the empty-penalties case would `AttributeError`
  at runtime, not return a non-Tensor).

**The instances (all proposing `-> Tensor`):**

| file:line | symbol | extra note |
|---|---|---|
| `penalty.py:11` | `Penalty.penalty` (abstract) | cross-file: imposes contract on all subclasses incl. research/; typing `x: Tensor` would ripple harder (PassThroughModule leaves `x` bare by convention) |
| `penalty.py:20` | `LambdaPenalty.penalty` | returns `self._penalty(x)`; `_penalty` is an arbitrary callable |
| `penalty.py:16` | `LambdaPenalty.__init__(penalty)` | natural `Callable[[Tensor], Tensor]`; could be too strict (caller might return a float scalar). Decide alongside the `.penalty` return so they agree |
| `l1_penalizer.py:90` | `SummedPenalties.penalty` | `int`-seeded accumulator; `-> Tensor` still safe; stricter fix seeds a zero Tensor |
| `penalty_mixins.py:12` | `ActsScaledByDecoderNormPenaltyMixin.penalty` | standalone mixin; super() unresolved |
| `penalty_mixins.py:32` | `LinearDecayPenaltyMixin.penalty` | standalone mixin; **note** the early `torch.zeros(1)` returns are not moved to `x.device/x.dtype` unlike `LinearDecayL1Penalty` — a latent device-mismatch bug, separate from the annotation |

Resolution for the cluster: annotate the *returns* only, and only if you intend
to type the whole `Penalty` family at once; leave `x` bare per the codebase
convention.

The remaining three findings in this subsystem are distinct:

### `src/saeco/components/penalties/penalty_mixins.py:5` — `ActsScaledByDecoderNormPenaltyMixin.__init__` (`self.decoder` attribute)
- Current: `self.decoder = None` then conditionally `self.set_decoder(decoder)`; checker infers `self.decoder: None` and flags `self.decoder.features[:]` in `penalty()`.
- Why thorny (**design inconsistency, not a type gap**): `set_decoder` stores `decoder.features.dec` (a `FeaturesView | None`), but `penalty()` then calls `self.decoder.features[:].norm(...)` — i.e. expects `.features`, which `FeaturesView` does **not** have. What `self.decoder` actually holds is inconsistent between this mixin and `L1PenaltyScaledByDecoderNorm`. The objects come from encoder/decoder modules (cross-file).
- Resolution: reconcile *what `self.decoder` stores* first; the annotation falls out of that. Design clarification, not a mechanical addition.

### `src/saeco/components/penalties/l0targeter.py:23` — `L0Targeting.get_value`
- Current: `def get_value(self): return self.value`.
- Issue: `self.value = scale` (default `1.0`), only updated by `*= 1 + incr` / `+= incr` (float incr), so stays float for default construction. `-> float` likely right.
- Why thorny: `scale` is an untyped ctor param; research callers pass it via `L0Targeting(...)`. If any caller passes a `Tensor` scale (plausible for a learnable/swept value), `self.value` and the return become `Tensor`. Cannot confirm float-ness without reading every research call site.
- Resolution: if adopted, tighten `scale: float` in the same pass so the `-> float` rests on something enforced.

### `src/saeco/components/hooks/feature_hooks.py:58` — `NormFeatures` (class parameterization)
- Current: `class NormFeatures(WrapsModule):` — bare, so the generic `WrapsModule[T: nn.Module = nn.Module]` defaults `T` to `nn.Module`.
- Issue: `NormFeatures` and `OrthogonalizeFeatureGrads` wrap a `HasFeatures` and access `self.__wrapped__.features`. Parameterizing would type `__wrapped__` precisely, paralleling `ResampledBias`/`EncoderBias` which use `WrapsModule[HasFeaturesModule]`.
- Why thorny (**Protocol vs nominal bound**): `WrapsModule` requires `T: nn.Module`, but the contract these wrappers need is `HasFeatures`, a `runtime_checkable` Protocol — *not* an `nn.Module` subclass. The repo comment on `WrapsModule` notes the lack of an intersection type forces this compromise. Using `WrapsModule[HasFeaturesModule]` (the concrete `HasFeatures & nn.Module` class) narrows the constructor's `wrapped: HasFeatures` param vs the type arg and may not match all call sites passing other `HasFeatures` modules.
- Resolution: deliberate design tradeoff (Protocol vs nominal bound); decide, don't auto-apply.

---

## components-resampling-jumprelu

This subsystem holds the two **clear quick wins** that were grouped here for
context, plus several **SweepableConfig** and **lazy-init / scalar-identity**
thorns.

### `src/saeco/components/jumprelu/kernels_fns.py:77` — `centrality` *(clear, confidence 0.78)*
- Current: `def centrality(kernel, centrality_measure=gauss):`
- The `Kernel` Protocol is defined in this same file; `integrate` already annotates `kernel: Kernel`, and `centrality` calls `kernel(x)`/`centrality_measure(x)` identically. Pure local addition. (Note: `centrality`/`check` are `__main__`-only diagnostics, so value is modest.)
- Resolution: `def centrality(kernel: Kernel, centrality_measure: Kernel = gauss):` — safe.

### `src/saeco/components/resampling/resampler.py:124` — `Resampler.assign_model` *(clear, confidence 0.83)*
- Current: `def assign_model(self, model: nn.Module) -> Self:` but body is `self.model = model; return model` — returns the arg, not `self`. pyright: "Module is not assignable to Self".
- All three call sites (runner.py:96, sae_architecture.py:150/151) **discard the return**, so changing the annotation to `-> nn.Module` is risk-free.
- The only judgement: `-> Self` may have encoded *intended* chaining that was never wired up. Fixing the annotation (to match the body) vs fixing the code (to return `self`) is the fork.
- Resolution: change to `-> nn.Module` unless chaining is wanted.

### `src/saeco/components/losses/losses.py:39` — `SparsityPenaltyLoss.loss`
- Current: override omits return; base declares `-> torch.Tensor`. Body seeds `l = 0` (int) then `l += sp_cache.sparsity_penalty.squeeze()`. With default `num_expected==1` `l` becomes Tensor, but `num_expected==0` would return the int `0`. pyright already flags the incompatible override.
- Why thorny: `-> torch.Tensor` is correct for every real usage (`num_expected>=1`) but technically narrows the `0`-return edge case.
- Resolution: decide whether the `num_expected==0` branch is reachable/supported before pinning `-> torch.Tensor`.

### `src/saeco/components/resampling/freq_tracker/ema.py:18` — `EMAFreqTracker.freqs`
- Current: `@property def freqs(self):` returns `self.activation_freqs`, which is `None` until the first `update_freqs`. Abstract base declares `-> torch.Tensor`; resampler consumers (`freqs[indices]`, `.freqs.shape[0]`, `.freqs < threshold`) assume a Tensor. pyright flags both the override and `zeros_like(None)` in `reset()`.
- Why thorny (**by-design lazy init**): two divergent fixes, each touching other code.
- Resolution: (a) widen to `torch.Tensor | None` — forces None-guards at every resampler call site and breaks the base's `-> torch.Tensor`; or (b) keep `-> torch.Tensor` and change construction to init `activation_freqs` to a tensor / fail-fast — a runtime/behaviour change.

### `src/saeco/components/resampling/anthropic_resampling.py:19` — `ModuleIO.inputs / outputs`
- Current: `inputs/outputs: dict[nn.Module, list[torch.Tensor]]`.
- Issue: the forward hook does `inputs[module].append(args)` where `args` is the positional-args **tuple**, and the consumer reads `res.inputs[enc_module][0][0]` (double index). So `inputs`' element type is `tuple[torch.Tensor, ...]`, not `torch.Tensor`. `outputs` stores the raw hook `output` (a single Tensor here, unverified across module types).
- Why thorny (**attrs field**): `@define` class fields; no converter/validator, so the change is purely static in practice — but per the attrs landmine, review as a field-type change. `outputs` element type depends on each hooked module's forward return.
- Resolution: tighten `inputs` to `list[tuple[torch.Tensor, ...]]`; confirm resample modules' return shapes before touching `outputs`.

### `src/saeco/components/resampling/resampler.py:59` — `ResamplerConfig.freq_balance_strat`
- Current: `freq_balance_strat: str = "sep"`. Only ever compared to `"mean"`/`"mix"` (with `"sep"` as the else) in `bias_freqbalance`/`wholistic_freqbalance` (lines 215-218, 279-282). Closed set — `Literal["sep","mean","mix"]` / StrEnum candidate.
- Why thorny (**SweepableConfig landmine**): annotation drives validation and is Swept-widened; narrowing could reject persisted/old sweep configs.
- Resolution: confirm no caller/sweep sets a value outside `{sep,mean,mix}`, then treat as a deliberate config-surface change.

### `src/saeco/components/resampling/anthropic_resampling.py:64` — `AnthResamplerConfig.enc_directions / dec_directions`
- Current: fields annotated `ResampleType` (an IntEnum) but defaulted to raw ints `0`/`1`, compared `== 0` (line 87) and used as list indices.
- Why thorny (**SweepableConfig; not a typing bug**): the int default is **by design** — pydantic coerces `0 -> ResampleType.enc_in`. Replacing defaults/comparisons with named members (`ResampleType.enc_in`/`.error`) is a readability/refactor-safety improvement that touches runtime config semantics.
- Resolution: leave as-is unless doing a deliberate readability refactor. Proposed: **(leave as-is)**.

---

## components-ops-misc

### `src/saeco/components/sae_cache.py:8` — `SAECache.sparsity_penalty`
- Current: `sparsity_penalty: torch.NumberType = ...`.
- Issue (**divergent Cache field declarations**): the same logical field is declared three ways — here `torch.NumberType`, in `core/reused_forward.py:11` as `float`, and in `trainer/train_cache.py:6` as bare `= ...`. penalty.py:9 writes a `Tensor`; losses.py:48 calls `.squeeze()` (Tensor API). `float` is clearly wrong; `Tensor` is closest.
- Why thorny (**Cache `= ...` sentinel, cross-file**): picking one type means touching at least two other files to keep declarations consistent.
- Resolution: likely `torch.Tensor`, but confirm whether the scalar reduction ever yields a python float vs a 0-dim Tensor; mirror across all three files.

### `src/saeco/components/sae_cache.py:7` — `SAECache.forward_reuse_dict`
- Current: bare `forward_reuse_dict: dict = ...` (duplicated in `core/reused_forward.py:10`).
- Issue: keyed by a `str`, stores a `ReuseForward` `output` that is module-dependent (`Tensor` or `list[Tensor]` per `core.module.c2t`).
- Why thorny (**Cache sentinel, heterogeneous value, cross-file**): `dict[str, Any]` is a free improvement, but tightening the value type requires auditing every `ReuseForward` output; the declaration is duplicated and must be mirrored (same divergence already seen on `sparsity_penalty`).
- Resolution: `dict[str, Any]` at minimum, mirrored in both files; tighter only after an audit.

### `src/saeco/components/ops/fnlambda.py:7` — `Lambda.__init__`
- Current: `def __init__(self, func, module=None):`.
- Issue: `func` is always called (`self.func(...)`), so `func: Callable` is safe. `module` is genuinely polymorphic — `forward` dispatches on `None`, `nn.Parameter`, `cl.Module`, or an arbitrary callable (final `self.module(x)` branch).
- Why thorny: `Lambda` is a core, widely-used op. A precise `module` type (`nn.Module | nn.Parameter | Callable[..., Any] | None`) is broad enough to add little safety, and the else-branch accepts a plain callable by design — pinning risks rejecting a legitimate caller.
- Resolution: annotate `func: Callable` (safe); leave `module` as-is or give it the full union deliberately.

---

## core

### `src/saeco/core/cache.py:465` — `Cache.__call__`
- Current: `def __call__(self, obj: T, force_watch=False) -> "SubCacher | T":` — always returns `SubCacher` at runtime, never `T`.
- Why thorny (**deliberate IDE-convenience lie**): the inline comment (466-467) states the typing is "not technically correct here but it gets the fields right in the IDE" — the `| T` union surfaces the wrapped object's attributes through the SubCacher proxy in editors.
- Resolution: narrowing to `-> SubCacher` is runtime-accurate but defeats the author's explicit autocomplete goal. UX tradeoff — author decides whether the ergonomics are still wanted. Proposed: **(leave as-is)**.

### `src/saeco/core/collections/propagator.py:40` — `Propagator._reduction / _cache_to_reduction / _binary_reduction / _binary_reduction_initial_value`
- Current: all initialized to `None`, reassigned in the (typed) `reduce()`. Checker infers them `None`, so `_reduce` reads like never-callable code.
- Why thorny (**None-as-falsey usage entangled with annotation choice**): `_reduce` raises if `_reduction is None`; `_binary_reduction`/`_cache_to_reduction` are read as plain truthiness even when None (`not self._binary_reduction`, `if self._cache_to_reduction`); and `(self._binary_reduction_initial_value is not None) + len(l)` does arithmetic on a bool. The "correct" optional types interact with these; the cleaner fix arguably sets `bool` defaults in `__init__` rather than `None` — a behavioural/design decision.
- Resolution: pick `... | None` annotations *or* change the defaults; coordinate with the falsey-usage sites.

### `src/saeco/core/module.py:16` — `Module.forward`
- Current: `def forward(self, x, *, cache: Cache, **kwargs):` — unannotated `x`, no return.
- Why thorny (**introspected `cache:` annotation + subclass-dependent return**): the `cache:` annotation is read by `saeco.components.type_acc_methods` (`_cache_types_from_annotation`/`_cache_parameter_and_types`) to derive cache dispatch — runtime-semantic. The return is genuinely subclass-dependent (`PassThroughModule` returns `x` unchanged; `Propagator` returns the reduction; basic_ops return tensors), so a single annotation is too loose (`Any`) or wrong for some override. Annotating `x` ripples to every override's input type.
- Resolution: leave to a human deciding the contract across the whole `Module` hierarchy. Proposed: **(leave / consider `x: torch.Tensor` + return)**.

---

## data-config

### `src/saeco/data/config/generation_config.py:21` — `DataGenerationProcessConfig.num_act_piles`
- Current: `def num_act_piles(self, num_tokens):`; body `return 1 + num_tokens // self.acts_per_pile`.
- Issue: the only caller passes ints, so `num_tokens: int -> int` matches usage. But `self.acts_per_pile` is the pydantic field `int | None` (default `2**15`), so `num_tokens // None` is a latent `TypeError` if it were ever None — the annotation can't be made fully clean without addressing that.
- Why thorny: the floor-division is unsound when `acts_per_pile` is None; a truly correct typing needs a runtime narrow/assert on a pydantic field.
- Resolution: (a) annotate only and accept the latent None path; or (b) also guard/narrow `acts_per_pile`.

### `src/saeco/data/config/data_cfg.py:130` — `DataConfig.acts_piler`
- Current: return already correctly `DictPiler`; bare-default params `write`, `target_gb_per_pile`, `num_tokens`.
- Why thorny (**scope decision on internal plumbing**): grouped with the cluster of untyped private-method params here (`_train_dataset`, `_get_databuffer`, `_get_queued_databuffer`, `train_data_batch_generator`, `get_split_tokens` with `model`/`batch_size`/`*_sites`/`nsteps`/`num_workers`/`queue_size`/`num_tokens`). Annotating is accurate but a how-far-to-annotate-internals call; **`target_gb_per_pile` is never read in the body**, hinting at dead/legacy intent — annotate-vs-remove.
- Resolution: stylistic/scope decision; lower priority than the clear return/Path findings. Resolve `target_gb_per_pile` (remove or keep) alongside.

### `src/saeco/data/config/locations.py:22` — `DataDirLocations.dir_prop`
- Current: `def dir_prop(m):` with an inner `wrapper(self, *args, **kwargs) -> Path`, stacked under `@property`.
- Why thorny (**generic decorator + descriptor composition**): a precise annotation needs ParamSpec/Concatenate over a bound-method-on-`DataDirLocations` composed with `@property` (the descriptor protocol) — finicky and easy to get subtly wrong. Low payoff: every consumer goes through the `@property`, which is independently typed `-> Path`.
- Resolution: invest in the full `def dir_prop[**P](m: Callable[Concatenate[DataDirLocations, P], Path]) -> ...` signature, or leave as-is.

---

## data-storage

The two `__getitem__` findings share a root cause: **the `Piler` index-domain
contract is unresolved**, and `DictPiler` defers to it.

### `src/saeco/data/piler/piler.py:196` — `Piler.__getitem__`
- Current: `def __getitem__(self, i) -> torch.Tensor:` with `return torch.cat(piles)  # type: ignore` at line 208.
- Issue: branches handle `int`/`list`/`slice` (callers pass all three).
- Why thorny (**a likely-dead/unsound branch, not just an annotation**): the final `else: piles = self.piles[i]` indexes a `GrowingDiskTensorCollection` whose `__getitem__` only accepts `str|int` and returns a single Tensor, so that branch is effectively dead/unsound and `torch.cat(piles)` on a single Tensor is masked by the `# type: ignore`. Pinning `i` precisely means deciding whether the else-branch is intended and whether the ignore can drop — a behavioural decision.
- Resolution: decide the index contract (`int | slice | list[int]`?) and the fate of the else-branch first.

### `src/saeco/data/piler/dict_piler.py:195` — `DictPiler.__getitem__`
- Current: `def __getitem__(self, i) -> DictBatch:` (return already declared). Internal callers pass `int` and `slice`; body forwards `i` unchanged to `Piler.__getitem__`.
- Why thorny: the accepted domain is whatever `Piler.__getitem__` accepts, which is itself unresolved (above). Pinning `int|slice` here may be narrower than `Piler` actually supports. *(Note: `future_annotations: true` for this file.)*
- Resolution: defer until the `Piler` index domain is settled, then mirror.

### `src/saeco/data/storage/disk_tensor_collection.py:43` — `MixedCache._remove_finalized_disktensor`
- Current: `def _remove_finalized_disktensor(self, key: str, disk_tensor: DiskTensor):`.
- Issue: `MixedCache` is generic over `DiskTensorType` (the popped value is `DiskTensorType`), so the bare `DiskTensor` is looser than the actual value.
- Why thorny (**Protocol conformance / contravariance**): this method exists to satisfy the `RemovableOnFinalize` Protocol (disk_tensor.py:67), which declares the param `DiskTensor[Any]`. Narrowing to `DiskTensorType` is contravariant tightening that could break structural conformance to that Protocol (and `DiskTensor.finalize()` stores `self` as a `RemovableOnFinalize` then calls `container._remove_finalized_disktensor(key, self)`).
- Resolution: decide whether to keep the Protocol-matching wide type or align *both* the Protocol and the method. Not a safe local change. (A `-> None` is the safe sub-fix.)

### `src/saeco/data/storage/disk_tensor.py:42` — `tensorclass`
- Current: `def tensorclass(dtype):` — maps a `torch.dtype` to a legacy typed-tensor constructor (`torch.FloatTensor`, ...).
- Why thorny (**dead code + awkward legacy types**): grep shows **no callers** anywhere. The `torch.*Tensor` names are legacy/quasi-deprecated; their precise static types are awkward (no clean common subtype beyond `type[torch.Tensor]`).
- Resolution: low value to annotate; the right move is arguably deletion — a behavioural decision.

---

## data-training

### `src/saeco/data/training_data/tokens_data.py:252` — `PermutedDocs.iter_docs_and_columns`
- Current: `def iter_docs_and_columns(self, batch_size, columns=None):`.
- Issue: `batch_size` is clearly `int`, `columns` is `list[str] | None` (normalized then iterated as keys) — safe params.
- Why thorny (**dynamic HF surface**): the yielded tuple's second element is `{col: ds[col] for col in columns}` from dynamic HF dataset row-slice indexing (`ds[col]` is `Any`); the first element's dtype depends on the dataset. A precise return can't be pinned without overcommitting. Also `get_docs`/`get_docs_and_columns` siblings are commented out, suggesting this is lightly-used scratch.
- Resolution: add the safe params; leave the return as `Iterator[tuple[torch.Tensor, dict[str, Any]]]` (no tighter). Verify callers before investing.

---

## trainer-core

This is the largest cluster. Several distinct threads:

**Thread A — `-> None` is the safe part, the param is the thorny part.** A run
of `Trainer` methods are called from exactly one or two in-repo sites with
tensor args, but the params they accept are wider than the call site (the
`loss()`/`forward()` consumers accept a polymorphic `Tensor | SAETrainBatch`,
and the cache hook-dispatch chain accepts `Cache | None` rather than the
concrete `TrainCache`). For all of these, `-> None` is a clean add; the param
narrowing is a mild contract-tightening to weigh.

| file:line | symbol | safe part | thorny part |
|---|---|---|---|
| `trainer.py:311` | `handle_backward` | `-> None`, `loss: torch.Tensor` | `cache: TrainCache` is narrower than `post_backward`'s `Cache | None` dispatch contract |
| `trainer.py:319` | `handle_step` | `-> None` | `cache: TrainCache` narrower than `post_step`'s `Cache | None` |
| `trainer.py:329` | `trainstep` | `-> None` | `x: torch.Tensor` tighter than `loss()` (which also accepts `SAETrainBatch`); `y: torch.Tensor | None` |
| `trainer.py:338` | `eval_step` | `-> None` | same polymorphic-`x` caveat as `trainstep` |
| `trainer.py:311` | `_train` (buffer typing) | `num_steps: int | None` | buffer element is really `SAETrainBatch` (`.input`/`.target`), but public `train()` annotates `Iterable[torch.Tensor] | None` — aligning both is cross-method and depends on what `get_databuffer()` yields |
| `trainer.py:165` | `Trainer.log` | `-> None` | `d` element type is not uniformly `float` (`logdict()` yields ints/tensors/strings; `loggables()` unverified; `mlog.log`'s accepted type not audited) — `dict[str, float]` too tight, `Mapping`/`dict[str, Any]` closer |

Resolution for Thread A: add the `-> None` returns now; decide the param
narrowing against the polymorphic `loss()`/`forward()` input and the
`Cache | None` dispatch contract.

**Thread B — `Trainable`'s `int`-seeded / polymorphic-`x` / shape-type cluster.**
These three are entangled and should move together:

### `src/saeco/trainer/trainable.py:124` — `Trainable.loss`
- `loss` accumulates `loss += m * coeffs.pop(...)` where `m` is a `torch.Tensor`, returning `loss`; natural return is `torch.Tensor`. But `loss` is seeded `int 0`, so the static type is `int | Tensor` (pyright flags `loss.item()`/`m * coeffs` at 140/141). `x` is intentionally polymorphic (`Tensor | SAETrainBatch`, unpacked at the top) so cannot be tightened.
- Thorny: the honest `-> torch.Tensor` needs either a cast or seeding `loss` as a tensor (a behavioural change), not a pure annotation.

### `src/saeco/trainer/trainable.py:166` — `Trainable.forward`
- Current `x: torch.Tensor` but the body branches on `isinstance(x, SAETrainBatch)` — so `x` is really `Tensor | SAETrainBatch`. Also the override is incompatible with `cl.Module/nn.Module.forward` (cache type + missing `**kwargs`), and the return is built from `self.dearrange(out, shape)` where `shape` is a `torch.Size`, not the `tuple[int,int]` dearrange declares.
- Thorny: fixing cleanly means touching the rearrange/dearrange shape types (below) and the base-class override complaint together.

### `src/saeco/trainer/trainable.py:158` — `Trainable.dearrange`
- Current `shape: tuple[int, int] | None`, but `rearrange` (line 149) returns `(x, shape)` where `shape` is `None` or `x.shape` (a `torch.Size`), which `tuple[int, int]` does not satisfy (pyright 171/228/233). Producer and consumer annotations disagree.
- Thorny: `torch.Size | None` is accurate, but `rearrange` has no return annotation; either (a) annotate `rearrange -> tuple[torch.Tensor, torch.Size | None]` and change `dearrange`, or (b) keep `tuple[int,int]` and cast in `rearrange`. A coordinated 2-method change on a torch dynamic surface.

Resolution for Thread B: treat `loss` / `forward` / `dearrange`+`rearrange` as
one coordinated change (seed/cast decision, polymorphic-`x` widening, and the
`torch.Size` vs `tuple[int,int]` reconciliation).

**Thread C — the SweepableConfig registry-key strings** (same landmine as
elsewhere: annotation drives validation + Swept widening; the closed set lives
cross-file in a registry):

| file:line | symbol | domain source |
|---|---|---|
| `train_config.py:58` | `l0_targeter_type: str = "gentle_basic"` | `l0targeter.TARGETER_TYPES` keys |
| `train_config.py:68` | `optim: str = "RAdam"` | `optim_config.get_optim_cls` accepted names (+ `"ScheduleFree"`, compared in `use_schedulefree`) |

Resolution: a human decides whether to introduce a StrEnum keyed off each
registry; keep members in sync; mind that `optim` is also compared `== "ScheduleFree"`.

**Remaining trainer-core findings:**

### `src/saeco/trainer/trainable.py:212` — `Trainable.get_acts`
- Current `def get_acts(self, x, cache: Cache = None, pre_acts=False) -> torch.Tensor:` — default `None` contradicts non-optional `Cache` (pyright 212). `@make_cache_optional` substitutes a real cache, so the body is non-None.
- Thorny: `cache: Cache | None` is the literal fix, but it's entangled — the **return is also wrong** (returns `acts`, or a 2-tuple `(acts, preacts)` when `pre_acts=True`, vs declared `-> torch.Tensor`, pyright 222/223), needing an overload or `-> torch.Tensor | tuple[...]`; and `cache.acts`/`cache.pre_acts` are dynamic Cache attributes (218/220).
- Resolution: fix `cache` default, return type, and dynamic-attr access together.

### `src/saeco/trainer/trainable.py:180` — `Trainable.param_groups`
- Current `def param_groups(self, optim_kwargs: dict) -> list[dict]:`. Keys are clearly `str`, values genuinely heterogeneous (str name, param lists, lr/betas/weight_decay).
- Thorny (**torch optimizer dynamic surface**): honest improvement is `dict[str, Any]`, only marginal over bare `dict`, and is what torch's own param-group API uses. Low value, mild over-tightening risk; borderline-skip.

### `src/saeco/trainer/trainer.py:206` — `Trainer.make_cache` (`cache.trainer = ...; cache.trainer = self`)
- Looks like a redundant assignment but is **load-bearing under Cache's write-once/watched-attribute semantics**: `... ` is the "declared but unset" sentinel; the next line fills it.
- Thorny: the typed fix is declaring `trainer: "Trainer"` on `TrainCache` (like the other `...` fields), adding a `Trainer` back-reference type that the Cache field machinery introspects (cross-file, runtime-semantic).
- Resolution: leave to a human; this is not dead code. Proposed: drop the placeholder and declare on `TrainCache`.

### `src/saeco/trainer/trainer.py:52` — `Trainer.__init__` (`save_callback`)
- `save_callback` is invoked as `self.save_callback(savename, save_weights=True, averaged_weights=...)` and `...(savename, save_weights=True, trainable=self.trainable)`. The real callee is `Architecture.save_to_path(self, path, save_weights, averaged_weights)`.
- Thorny: a precise Protocol would **surface an existing kwarg mismatch** — `save()` passes `trainable=` (not declared on `save_to_path`) and `averaged_weights=` a state_dict where `save_to_path` types it `bool|None`. Tightening creates new errors reflecting a real latent inconsistency, not a clean win.
- Resolution: `Callable[..., None] | None` is a safe-but-weak improvement; a precise Protocol needs `save_to_path` reconciled first.

### `src/saeco/trainer/trainer.py:149` — `Trainer.subject_model`
- `@property` returns `self.cfg.data_cfg.model_cfg.model`.
- Thorny (**nnsight dynamic surface**): the concrete type is a transformer/nnsight `LanguageModel` surface (not audited); annotating couples this file to an nnsight dynamic type.
- Resolution: skip-or-thorny; do not guess.

---

## trainer-aux

Threads here: **nnsight proxies**, the **scalar-identity-vs-Tensor** normalizer
convention, **research PID controller knobs**, and a couple of
**dead/deprecated/stub** surfaces.

**Scalar-identity-vs-Tensor (normalizers).** Two findings share one convention
question: several normalizer methods return a python scalar (`0`/`1`/`1.0`) in a
"DONTUSE"/identity branch but are annotated as returning a `Tensor`. The scalars
are *intentional broadcasting identities* (`x - self.mu_s(x)`, `x / self.std_e(x)`).
The honest union (`Float[Tensor, ...] | int`/`| float`) is awkward and adds
little checker value; the real decision is a **project-wide convention** (allow
scalar identities in the API, or force `torch.zeros_like`/`ones_like`).

- `normalizer.py:442` — `GeneralizedNormalizer.mu_s` returns int `0` in DONTUSE (vs `Float[Tensor, "batch 1"]`). `std_s` already documents this as `... | float`; `mu_e` returns `0`, `std_e` returns `1`. Buffers (`_mu_s`/`_mu_e`) can also be uninitialized NaN tensors. Decide a consistent convention across `mu_s`/`std_s`/`mu_e`/`std_e` first.
- `normalizer.py:113` — `AffineNormalizer.shift -> Tensor`, but `L2Normalizer.shift` (150-151) returns int `0` (deliberate no-shift identity). Same convention question.

**Research PID controller knobs (`L0Targeter`).** Two entangled findings:

- `l0targeter.py:294` — `L0Targeter.__call__(self, l0, t) -> float:` returns `.item()` only inside the `mode == 'step'/'clamp'/'clamp_all'` branches; any other mode **returns `None`**, contradicting `-> float`. The cleanest fix is a *runtime* one (final `raise`/exhaustive handling), since callers (the PID loop) treat the result as a number.
- `l0targeter.py:231` — `self.mode = "clamp_all"` compared against the closed set `{'step','clamp','clamp_all'}`. A `Literal`/StrEnum would catch typos and tie off the `__call__` None-leak — but `mode`, `velocity_mode`, `direct_mode`, `by_sign`, etc. are experimental knobs set/mutated ad hoc on an unstable research surface with no API contract; a `Literal`/StrEnum is a design decision that risks the experiment knobs.

Resolution: handle the None-leak (runtime) and the `mode` Literal together, with
the author, treating it as research-surface design.

**Remaining trainer-aux findings:**

### `src/saeco/trainer/optim_config.py:5` — `get_optim_cls`
- Returns a value from a dict mixing concrete optimizer classes (Adam/SGD/...), the third-party `AdamWScheduleFree` class, and a `lambda *a, **k` factory (`AmsgradW`); callers immediately call the result.
- Thorny: no single tight return is fully accurate. `type[Optimizer] | Callable[..., Optimizer]` is the honest union, but `AdamWScheduleFree`'s MRO needs checking (may not subclass `torch.optim.Optimizer` cleanly); `Callable[..., Any]` undersells it.
- Resolution: decide how strict to be about the schedulefree class.

### `src/saeco/trainer/recons.py:53` — `get_recons_loss`
- An **existing Protocol already specifies the exact shape**: `evaluation_protocol.py:9 ReconstructionEvaluatorFunctionProtocol` declares `cast_fn: Callable[[], AbstractContextManager] = ...`, `data_cfg: DataConfig`, `-> dict[str, float]`. `get_recons_loss` (and `with_bos`/`no_bos` at 10/31) implement it but are looser: no return, `data_cfg: DataConfig = None` (asserted not-None at 63), untyped `cast_fn=...`, `tokens=None` (used unconditionally at 83). The returned dict's values are all floats, so `dict[str, float]` is correct.
- Thorny (**cross-file ripple + sentinel-default**): removing `data_cfg = None` is a signature/call change across the wrappers + the architecture call site; `cast_fn=...` uses Ellipsis as a sentinel and is invoked as `with cast_fn():` — typing it `Callable[[], AbstractContextManager]` matches the protocol but the Ellipsis default is not callable, so any caller relying on the default crashes (it's an implicit required arg).
- Resolution: coordinate the three recons functions + protocol as a set.

### `src/saeco/trainer/recons.py:182` — `to_losses`
- Wraps a runner closure (`with_sae_runner`/`zero_ablated_runner`/...) returning `model.output.logits.save()`, and returns a `runner(prepared: SubjectBatchInputs)` yielding the cross-entropy Tensor.
- Thorny (**nnsight proxy lifecycle**): `model_callable(prepared)` returns an nnsight proxy that behaves like a Tensor only after the trace context exits (and `out[:, :-1]` indexes it). Annotating the runner return `Tensor` is useful for callers but technically lies about the proxy lifecycle.
- Resolution: only worth it if the team is comfortable treating saved proxies as Tensors at the type level.

### `src/saeco/trainer/saved_model_source_info.py:20` — `ModelReloadInfo.source_backup`
- Current `source_backup: str = None` — declared `str`, defaults `None`; some construction paths never set it.
- Thorny (**pydantic field on a @deprecated model**): annotation drives validation; the class is `@deprecated` (migration to `ArchReloadInfo`) and still deserialized via `model_validate_json` in runner.py, so widening to `str | None` could shift validation on stored JSON. Low value given deprecation.
- Resolution: decide whether to touch a deprecated model at all. (`str | None` is the literally-correct fix.)

### `src/saeco/trainer/scheduler.py:19` — `Scheduler.__init__`
- `Scheduler.__init__` has an empty (`...`) body, takes typed `schedule: Schedule`, stores nothing. Grep shows **no callers** — a stub.
- Thorny (**not a typing fix**): the `Schedule`/`RunSchedule`/`ResampleSchedule` SweepableConfig models are already correctly typed (`dict[int | RunFloat, float]`).
- Resolution: a human decides whether to delete or implement, not annotate. Proposed: **(none — likely dead/placeholder)**.

---

## misc-sweeps

A grab-bag, but three are **latent runtime bugs surfaced by the audit** (not
pure annotation gaps), one is an **import cycle**, and two are **attrs/legacy**.

### `src/saeco/misc/exception_location_hint.py:4` — `locate_cache_exception` (cache param)
- Body reads `cache._name` (a `str` on `saeco.core.cache.Cache`); the only caller passes a `Cache`, so the type is `Cache`.
- Thorny (**import cycle**): `core/cache.py` imports this module at top level (cache.py:7), so a runtime `from saeco.core.cache import Cache` here is circular.
- Resolution: use a `TYPE_CHECKING`-guarded import + string/`from __future__` annotation; otherwise leave untyped.

### `src/saeco/misc/paths.py:4` — `SAVED_MODELS_DIR`
- `os.environ.get("SAECO_SAVED_MODELS_DIR", Path.home() / "...")` infers `str | Path` (set env var yields `str`). The only consumer (trainer.py:393 `save_dir / sweep_name / run_name`) uses `/`, which only works on `Path`.
- Thorny (**latent runtime bug, not just typing**): when the env var is actually set this breaks at runtime. Annotating `: Path` would *lie* when the env var is set; the honest fix wraps the expression in `Path(...)` — a behavioural change to a cross-file module constant.
- Resolution: decide whether to coerce to `Path` at definition.

### `src/saeco/sweeps/sweep_data.py:60` — `SweepData.save / sweep_id`
- `sweep_id: str | None` (pydantic), but `save()` line 60 assigns `self.get_sweep_number()` (returns `int`) on one branch and string-concats (`self.sweep_id += str(...)`) on the other — the branches disagree on runtime type, and without `validate_assignment` the field can actually hold an int contradicting its annotation.
- Thorny (**logic inconsistency on a pydantic field, not just typing**): fixing requires choosing a single representation (likely `str`) and possibly adjusting `get_sweep_number`'s return and its other caller (line 62).
- Resolution: design decision; pick a representation, not a safe annotation tweak.

### `src/saeco/mlog/mlog_.py:43` — `mlog.create_sweep`
- Forwards to `self.logger_instance.sweep(...)`, but the wired logger `WandbCustomLogger` (the only entry in `LOGGER_CLASSES`) defines **no `sweep` method** (only `CustomSweeper` has a stub). So a return annotation can't be honestly added — the method would `AttributeError` if ever called against the real logger. `project` is also un-annotated.
- Thorny (**latent runtime bug surfaced by the audit**): the meaningful fix is reconciling the logger interface (add `sweep` to `WandbCustomLogger` or remove `create_sweep`); annotating `project: str | None` is safe but cosmetic.
- Resolution: flag for a human — interface reconciliation, not an annotation.

### `src/saeco/mlog/fns.py:316` — `BaseLogger._sweep_inst_config`
- Current `_sweep_inst_config = None` inside an `@define` (attrs) class. Unannotated, so it is a **plain class attribute**, not a field; later reassigned to a dict.
- Thorny (**attrs field-collection landmine**): adding `: dict | None` both types it *and* converts it into a real attrs field (init param / slot), altering construction semantics — a runtime change.
- Resolution: decide whether it should be an attrs field (annotate) or stay a class-level sentinel (`field(default=None)` or as-is).

### `src/saeco/mlog/fns.py:30` — `CustomSweeper.log / .agent / .sweep` (callbacks & params)
- `CustomSweeper` mirrors `WandbCustomLogger` (whose equivalents ARE typed: `data: dict`, `step: int | None`, `sweep_index: int`, `sweep_hash: str`, `function()`), so intended types are knowable by analogy + the `mlog_.py` callers.
- Thorny (**unused/legacy parallel implementation**): `LOGGER_CLASSES` only wires `WandbCustomLogger`, so `CustomSweeper` is not instantiated by the registry; `function` is a zero-arg callback whose precise type (`Callable[[], object]` vs `Callable[..., Any]`) is a judgement call.
- Resolution: triage whether `CustomSweeper`/`BaseLogger` should be kept at all before investing.

---

## Suggested triage order

**Quick wins (low risk, mostly mechanical, do first):**
- The two **`clear`** findings: `kernels_fns.py:77 centrality` and
  `resampler.py:124 assign_model` (the only fork there is annotation-vs-chaining,
  and all callers discard the return).
- The **`-> None` returns** across Thread A in trainer-core (`handle_backward`,
  `handle_step`, `trainstep`, `eval_step`, `_train`'s `num_steps`, `Trainer.log`):
  add the returns now, defer only the param narrowing.
- The safe sub-fixes called out per finding: `d: dict[str, Any]`
  (`from_json`), `-> str` (`get_src`), `func: Callable` (`Lambda`),
  `dict[str, Any]` (`forward_reuse_dict`, `param_groups`), `-> None`
  (`_remove_finalized_disktensor`), the `tokens_data` params.

**Single-decision-then-mechanical (medium):**
- The **`Penalty` family return contract** — one decision ("type the family,
  returns only, leave `x` bare"), then apply to all six instances at once.
- The **normalizer scalar-identity convention** (`mu_s`/`std_s`/`mu_e`/`std_e`,
  `AffineNormalizer.shift`) — pick the convention, then apply.
- The **paramsight return annotations** (`get_arch_config_class`,
  `from_json` return) — one decision on accepting/suppressing the precedented
  `reportReturnType` noise.
- The **dead/stub triage** (`tensorclass`, `Scheduler`, `CustomSweeper`/`BaseLogger`,
  `geo_med`, `data_cfg.target_gb_per_pile`): each is a delete-vs-keep call, not
  an annotation; resolving them removes findings outright.

**Real design decisions (do last, may stay open):**
- The **SweepableConfig stringly-typed fields** (`b2_technique`,
  `freq_balance_strat`, `l0_targeter_type`, `optim`; plus the `enc/dec_directions`
  enum-default readability question) — each is a validation-surface change that
  must confirm no persisted/old config breaks.
- The **Cache field-divergence** (`sparsity_penalty`, `forward_reuse_dict`,
  `cache.trainer`) — needs a single chosen type mirrored across files under
  watched-attribute semantics.
- The **`Piler` index-domain contract** (gates `Piler.__getitem__` *and*
  `DictPiler.__getitem__`, and the dead else-branch / `# type: ignore`).
- The **`Trainable` cluster** (`loss`/`forward`/`dearrange`+`rearrange`/`get_acts`)
  — coordinated, with seed/cast and `torch.Size` reconciliation, plus the
  base-class override compatibility.
- The **latent runtime bugs** (`SAVED_MODELS_DIR` str/Path, `create_sweep`'s
  missing `sweep` method, `SweepData.sweep_id` int/str divergence,
  `save_callback`'s kwarg mismatch, `num_act_piles`' None floor-division) —
  fix the behaviour, not the annotation.
- The **structural/Protocol reconciliations** (`SupportsResample`,
  `MixedCache`/`RemovableOnFinalize`, `WrapsModule[HasFeaturesModule]` Protocol
  vs nominal bound, `ActsScaledByDecoderNormPenaltyMixin.decoder` "what does it
  hold" question).
- The **deliberate lies / dynamic surfaces** to leave as-is unless re-litigated:
  `Cache.__call__`'s IDE-convenience `| T`, `Module.forward`'s introspected
  `cache:`, the nnsight surfaces (`subject_model`, `to_losses`), and the
  research PID knobs (`L0Targeter.mode`/`__call__`).
