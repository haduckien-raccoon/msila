## Basic Decoder

`models/basic_decoder.py` is the minimal dense-prediction head used
during Day-1 Architecture QA.

Its purpose is to verify that the fused DINOv3 representation can be
converted into a full-resolution anomaly map.

### Interface

Input:

    Z : [B,C,h,w]

Processing:

    Conv3x3
        ↓
    GELU
        ↓
    Conv1x1
        ↓
    [B,1,h,w]
        ↓
    bilinear interpolation

Output:

    anomaly_logits : [B,1,H,W]

where H,W are the spatial dimensions of the original input image.

### Responsibility

The decoder is responsible for:

1. transforming the fused representation into one anomaly logit per
   spatial location;
2. reducing the feature channels from C to 1;
3. restoring the prediction to the original image resolution.

### Non-responsibilities

The Day-1 decoder does NOT implement:

- multi-scale feature fusion;
- FPN;
- U-Net skip connections;
- attention;
- Local/Context interaction;
- anomaly thresholding;
- sigmoid activation;
- post-processing.

These components are intentionally excluded so that Day-1 tests only
verify the architectural forward path.

### Tensor contract

`contracts.py` verifies the decoder boundary.

Before decoding:

    [B,C,h,w]

must be a valid finite floating-point feature tensor.

After decoding:

    [B,1,H,W]

must satisfy:

- same batch size as input image;
- exactly one output channel;
- same H,W as input image;
- no NaN;
- no Inf.

`contracts.py` performs validation only. It never resizes or modifies
the tensors.