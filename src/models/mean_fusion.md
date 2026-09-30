## Mean Fusion Baseline

`models/mean_fusion.py` implements the parameter-free fusion baseline
used in Day 1 Architecture QA.

Given three adapted DINOv3 features:

    A4  : [B,C,h,w]
    A8  : [B,C,h,w]
    A12 : [B,C,h,w]

the fused representation is

    Z = (A4 + A8 + A12) / 3

and therefore

    Z : [B,C,h,w]

### Responsibility

MeanFusion is responsible only for aggregating already compatible
feature maps.

It does not:

- resize feature maps;
- change channel dimensions;
- perform feature projection;
- perform spatial alignment;
- learn fusion weights;
- apply attention;
- implement Local/Context fusion.

Mean Fusion is a control baseline, not an MS-ILA novelty module.

### Tensor-contract interaction

Before fusion, `contracts.py` verifies that:

    b4, b8, b12

all satisfy:

    [B,C,h,w]

and have:

- identical tensor shapes;
- identical dtype;
- identical device;
- finite values;
- exactly the expected feature keys.

If these conditions are violated, MeanFusion must fail instead of
silently modifying the features.

After fusion, `contracts.py` verifies that the output still satisfies:

    [B,C,h,w]

Thus the responsibility is separated as:

    contracts.py
        ↓
    verify interface

    mean_fusion.py
        ↓
    perform fusion

    contracts.py
        ↓
    verify output