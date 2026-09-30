## Tensor Contract — MS-ILA

`models/contracts.py` defines the tensor interface shared by all
MS-ILA modules.

### Purpose

The contract prevents silent tensor-layout and shape mismatches between:

DINOv3 → Adapter → Fusion → Decoder.

It performs validation only. It does not perform feature projection,
interpolation, alignment, or model computation.

### Canonical tensor layout

All dense feature maps inside MS-ILA use:

    [B, C, H, W]

The pipeline must not mix `[B,N,C]` transformer tokens and
`[B,C,H,W]` image feature maps across modules.

The DINOv3 extractor is responsible for converting patch tokens into
the canonical BCHW representation before returning its features.

### Day-1 interfaces

Input image:

    [B, 3, H, W]

DINOv3 output:

    b4  -> [B, C, h, w]
    b8  -> [B, C, h, w]
    b12 -> [B, C, h, w]

For the Day-1 Mean Fusion baseline, the three features must have
identical shapes.

Residual Adapter:

    [B,C,h,w] -> [B,C,h,w]

Mean Fusion:

    {b4,b8,b12} -> [B,C,h,w]

Basic Decoder:

    [B,C,h,w] -> [B,1,H,W]

### Validation responsibilities

`contracts.py` checks:

1. Tensor dimensionality is BCHW.
2. Input image contains three RGB channels.
3. DINO output contains exactly b4, b8 and b12.
4. Feature tensors contain no NaN/Inf.
5. Features use consistent device and dtype.
6. Day-1 features have identical shapes before Mean Fusion.
7. Fused feature preserves the expected shape.
8. Decoder returns `[B,1,H,W]`.

### Non-responsibilities

`contracts.py` MUST NOT:

- resize feature maps;
- perform interpolation;
- perform channel projection;
- implement feature fusion;
- implement Local/Context branches;
- train model parameters.

Spatial/channel alignment will be implemented explicitly in the
Day-2 projection/alignment modules.