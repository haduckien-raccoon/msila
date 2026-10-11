# D8-TV2 — Fusion API handoff

```python
from src.models.fusions import WeightedSumFusion, ConcatFusion, GatedFusion

fusion = WeightedSumFusion(dim=64)  # or ConcatFusion / GatedFusion
fused = fusion(features)           # Tensor [B, 64, H, W], not a trace dictionary
```

`dim` comes from the resolved `fusion.dim`; it is independent of backbone C
and Adapter r/d. All three modules support any positive integer width. Move the
module and inputs to the same device/dtype as with ordinary PyTorch modules.
`validate_input=True` is the default and reuses `validate_multiview_features`.

`features` is a mapping with exactly these six already aligned/projected tensors
of identical floating dtype, device and [B, dim, H, W] shape, in logical order:

1. `local_b4`
2. `local_b8`
3. `local_b12`
4. `context_b4`
5. `context_b8`
6. `context_b12`

Dictionary insertion order does not affect source ordering. The existing
`validate_g2_fused_feature` can validate the returned tensor at integration.

| Contract ID | Import | Parameters | Diagnostic API |
| --- | --- | --- | --- |
| `weighted_sum` | `WeightedSumFusion` | 6 global scalar logits | `normalized_weights()` returns [6] |
| `concat` | `ConcatFusion` | 6*dim² + dim (Conv1x1 with bias) | `projection` is the Conv2d |
| `gated` | `GatedFusion` | 6*(dim + 1) (six independent C->1 Conv1x1 gates) | `normalized_weights(features)` returns [B,6,H,W] |

Weighted and Gated initialize to uniform weights. Concat retains standard
PyTorch Conv2d initialization. Gated normalizes sigmoid gates in log space,
preserving sum-to-one weights even when all sigmoid values would underflow.
Weights remain differentiable; each module supports ordinary `state_dict`
checkpoint save/load.

TV1 owns the factory, label aliases and E5 integration. Merge this TV2 handoff
into the integration branch before D9 factory acceptance. D8 verifies code on
CPU; it supplies no training results or AU-PRO ranking.
