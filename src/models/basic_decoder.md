Fused feature
[B,C,h,w]
    │
    ▼
Conv 3×3
C → C/2
    │
    ▼
GELU
    │
    ▼
Conv 1×1
C/2 → 1
    │
    ▼
[B,1,h,w]
    │
    ▼
Bilinear upsample
    │
    ▼
[B,1,H,W]