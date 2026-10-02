# E4 — Parameter Count Protocol

**Task:** E4 only  
**Implementation:** `src/eval/efficiency.py`  
**Outputs:** `params.json` per candidate  
**Not implemented here:** latency (E5), peak VRAM (E6)

## 1. Scientific definition

For each unique registered PyTorch parameter \(p\), its scalar count is

\[
N(p)=\operatorname{numel}(p).
\]

The report uses

\[
N_{\text{total}}=\sum_{p\in\Theta}N(p)
\]

and

\[
N_{\text{trainable}}
=\sum_{p\in\Theta,\;p.\mathrm{requires\_grad}=\mathrm{True}}N(p).
\]

`torch.numel()` is exactly the number of elements in a tensor. PyTorch's
`named_parameters()` also supports duplicate removal; therefore a shared/tied
`nn.Parameter` object must be counted **once**, not once for every module path.

E4 deliberately does **not** count buffers, activations, gradients, optimizer
states, cached DINO features, latency, or VRAM. Those are different quantities.

`trainable_parameters` here means `requires_grad=True`. It does not inspect
whether a custom optimizer intentionally omitted a parameter.

## 2. Why the scope field is mandatory

The project has `CachedFeatureTrainingModel`. Its training path starts from
cached frozen DINO features, so **DINOv3 is not instantiated inside that model**.

Therefore:

```text
scope = cached_trainable_pipeline
```

means:

```text
Adapter -> Alignment -> Projection -> Fusion -> Decoder
```

and its `total_parameters` must **not** be described as the total parameter
count of the full deployment pipeline.

If the frozen DINO backbone is instantiated inside another full inference
model, count that exact model separately with:

```text
scope = full_inference_model
```

This distinction prevents an artificially small "total params" claim.

## 3. Manual formula for the current MS-ILA cached pipeline

Notation:

- \(C\): frozen DINO feature channels
- \(b\): adapter bottleneck channels
- \(k\): depthwise kernel size
- \(d\): fusion dimension
- \(B=3\): DINO blocks 4, 8, 12
- \(h=\max(\lfloor d/2\rfloor,1)\): BasicDecoder hidden channels

### 3.1 One ResidualAdapter2d

Current adapter:

```text
1x1 down projection
depthwise kxk convolution
1x1 up projection
learnable gamma
```

Hence

\[
N_{\text{adapter}}
=(Cb+b)+(bk^2+b)+(bC+C)+1
\]

or

\[
N_{\text{adapter}}
=2Cb+bk^2+2b+C+1.
\]

The current cached model has one adapter for each DINO block and shares that
block adapter between Local and Context, so

\[
N_{\text{adapters}}=3N_{\text{adapter}}.
\]

### 3.2 Context alignment

`ContextToLocalAligner` uses geometry/grid sampling and has no learnable
`nn.Parameter`:

\[
N_{\text{aligner}}=0.
\]

### 3.3 Feature projection

With the current default `share_across_views=True`, each DINO block has one
\(1\times1\) projection \(C\to d\), including bias:

\[
N_{\text{projection}}=3(Cd+d).
\]

If Local and Context use separate projectors:

\[
N_{\text{projection}}=6(Cd+d).
\]

### 3.4 Attention fusion

Current `AttentionFusion` contains:

- bias-free `Linear(d,1)`: \(d\) parameters;
- six source priors: \(6\) parameters.

Therefore

\[
N_{\text{fusion}}=d+6.
\]

### 3.5 Basic decoder

The decoder is:

```text
Conv3x3(d -> h, bias=True)
Conv1x1(h -> 1, bias=True)
```

Therefore

\[
N_{\text{decoder}}
=(9dh+h)+(h+1)
=9dh+2h+1.
\]

### 3.6 Closed-form total

For the default cached MS-ILA pipeline:

\[
N_{\text{cached}}
=
3N_{\text{adapter}}
+
N_{\text{projection}}
+
N_{\text{fusion}}
+
N_{\text{decoder}}.
\]

The code implements this independently in
`manual_cached_msila_parameter_formula()`. This is the **manual/reference side**
of the E4 PASS gate; it does not traverse `model.parameters()`.

## 4. Usage for one candidate

Example for a candidate using reduction \(r=4\), fusion dimension \(d=128\):

```python
from src.eval.efficiency import verify_cached_msila_parameter_count
from src.models.cached_training import CachedFeatureTrainingModel

model = CachedFeatureTrainingModel.build_default(
    in_channels=384,
    fusion_dim=128,
    adapter_reduction=4,
    adapter_kernel_size=3,
    share_projection_across_views=True,
)

result = verify_cached_msila_parameter_count(
    model,
    candidate_id="r4_d128",
    in_channels=384,
    fusion_dim=128,
    adapter_reduction=4,
    adapter_kernel_size=3,
    share_projection_across_views=True,
    output_path="outputs/day04/r4_d128/params.json",
)

print(result["status"])
print(result["report"]["totals"])
```

For these **example dimensions only**:

\[
b=384/4=96
\]

and the closed-form cached-pipeline count is:

```text
Adapter / block : 75,169
3 adapters      : 225,507
Projection      : 147,840
Fusion          : 134
Decoder         : 73,857
--------------------------------
Total           : 447,338
```

This number is an arithmetic example, **not a claimed final candidate result**;
the actual candidate configuration must be supplied to the function.

## 5. Output contract

`params.json` records at least:

```json
{
  "candidate_id": "r4_d128",
  "scope": "cached_trainable_pipeline",
  "totals": {
    "total_parameters": 447338,
    "trainable_parameters": 447338,
    "frozen_parameters": 0
  }
}
```

The real file also includes every unique parameter's name, shape, number of
elements, `requires_grad`, dtype/device, and aliases caused by parameter sharing.
This makes the count auditable instead of reporting only one unexplained number.

## 6. Hard PASS criterion

For the current cached MS-ILA model, E4 is PASS only when all are true:

```text
actual total params       == closed-form total
actual trainable params   == closed-form trainable total
Adapter component count   == manual formula
Projection count          == manual formula
Fusion count              == manual formula
Decoder count             == manual formula
unexpected parameter count == 0
```

A mismatch raises `ParameterCountError`; it is never silently rounded or
accepted.

For a generic model where no architecture-specific formula exists, use
`parameter_report()` plus `assert_parameter_count()` with a separately derived
manual count.

## 7. Scientific interpretation

Parameter count measures **model capacity/size**, not inference cost.

Therefore E4 supports statements such as:

> Candidate A introduces X additional trainable scalar parameters relative to
> Candidate B.

E4 alone does **not** justify:

```text
A is faster
A uses less peak VRAM
A has lower FLOPs
A is more accurate
```

Those require E5/E6 or the accuracy evaluator.

## 8. Evidence

The implementation follows PyTorch's parameter model:

- `torch.numel(tensor)` returns the total number of scalar elements.
- `nn.Module.named_parameters(..., remove_duplicate=True)` removes duplicated
  parameter objects by default.
- `requires_grad` specifies whether autograd records operations for that tensor.

Project-specific formulas above are derived directly from the current modules:

```text
src/models/residual_adapter.py
src/models/feature_projection.py
src/models/attention_fusion.py
src/models/basic_decoder.py
src/models/cached_training.py
```

No FLOP, latency, or memory estimate is mixed into this count.
