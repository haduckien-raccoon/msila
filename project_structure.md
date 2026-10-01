project/
├── data/
│   ├── multiview_transform.py
│   └── multiview_transform.md
│
├── geometry/
│   ├── view_meta.py
│   └── view_meta.md
│
├── models/
│   ├── dinov3_extractor.py
│   ├── dinov3_extractor.md
│   ├── adapter.py
│   ├── fusion.py
│   └── decoder.py
│
└── tests/
    ├── test_multiview_transform.py
    ├── test_view_meta.py
    └── test_dinov3_extractor.py



data/
    xử lý dữ liệu đầu vào
    crop / resize / normalize / augmentation

geometry/
    metadata hình học
    coordinate mapping
    transform matrix

models/
    các nn.Module
    DINOv3 backbone/extractor
    adapter
    fusion
    decoder

tests/
    unit test tương ứng với từng module