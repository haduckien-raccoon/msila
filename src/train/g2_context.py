"""TV1 paired-view boundary and native E4 inference; no synthetic generation.

Wrap G1's exact tile index/order. Both crops come from one existing native
sample, using the checked tiling and geometry modules (including edge padding).
"""
from dataclasses import asdict, replace

import torch
from torch.utils.data import Dataset

from src.data.loader import g1_tile_collate, normalize_dinov3
from src.data.tiling import (crop_with_padding, extract_local_context,
                             generate_tile_records, stitch_tiles_hann)
from src.geometry.view_meta import build_view_meta


def paired_record(record, context_size=768):
    x0, y0, x1, y1 = record.local_xyxy
    if x1 - x0 != 512 or y1 - y0 != 512:
        raise ValueError("E4 requires the unchanged 512px G2 Local tiles")
    if type(context_size) is not int or context_size < 512 or (context_size - 512) % 2:
        raise ValueError("Context FOV must be an even integer >= 512")
    margin = (context_size - 512) // 2
    return replace(record, context_xyxy=(x0-margin, y0-margin, x1+margin, y1+margin))


def prepare_pair(image, record, *, sample_id=None):
    """Normalized views plus pixel-edge geometry, even for padded native edges.

    Translate both boxes into a virtual padded canvas for build_view_meta's
    bounds checks. Their common translation cancels in Local->Context; crop
    pixels still use the original native boxes and existing padding policy.
    """
    local, context = extract_local_context(image, record)
    h, w = image.shape[-2:]
    lx0, ly0, lx1, ly1 = record.local_xyxy
    cx0, cy0, cx1, cy1 = record.context_xyxy
    left, top = max(0, -cx0, -lx0), max(0, -cy0, -ly0)
    right, bottom = max(0, cx1-w, lx1-w), max(0, cy1-h, ly1-h)
    shift = lambda box: (box[0]+left, box[1]+top, box[2]+left, box[3]+top)
    geometry = build_view_meta(
        source_hw=(h+top+bottom, w+left+right),
        local_box_xyxy=shift(record.local_xyxy), context_box_xyxy=shift(record.context_xyxy),
        local_input_hw=(512, 512), context_input_hw=(512, 512),
    ).as_tensor_dict()
    meta = dict(geometry=geometry, native_hw=(h, w),
                local_native_xyxy=record.local_xyxy, context_native_xyxy=record.context_xyxy,
                padding_ltrb=(left, top, right, bottom), native_sample_id=sample_id,
                local_sample_id=sample_id, context_sample_id=sample_id)
    return normalize_dinov3(local), normalize_dinov3(context), meta


class PairedG2Tiles(Dataset):
    def __init__(self, tiles, *, context_size=768):
        self.tiles, self.context_size = tiles, context_size
        # Validate before a worker starts loading data.
        if len(tiles):
            paired_record(tiles.records[0][1], context_size)

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        native_index, record = self.tiles.records[index]
        sample = self.tiles.native[native_index]  # one native synthetic, both views
        record = paired_record(record, self.context_size)
        local, context, view_meta = prepare_pair(sample["image"], record,
                                                  sample_id=sample["meta"]["sample_id"])
        mask = crop_with_padding(sample["mask"], record.local_xyxy, pad_mode="constant")
        return dict(image=local, context=context, mask=mask, view_meta=view_meta,
                    meta={**sample["meta"], "tile": asdict(record)})


def paired_tile_collate(samples):
    batch = g1_tile_collate(samples)
    batch.update(context=torch.stack([sample["context"] for sample in samples]),
                 view_meta=[sample["view_meta"] for sample in samples])
    return batch


def paired_model_forward(model, batch):
    return model(batch)


@torch.no_grad()
def predict_native_e4(model, image, cfg, device):
    """Sigmoid tile maps, Hann stitched directly into the full native canvas."""
    model.eval()
    h, w = image.shape[-2:]
    records = generate_tile_records(h, w, cfg["data"]["tile_size"], cfg["data"]["overlap"],
                                    context_size=cfg["paired_views"]["context_size"])
    maps = []
    size = cfg["evaluation"]["tile_batch_size"]
    for start in range(0, len(records), size):
        views = [prepare_pair(image, record) for record in records[start:start+size]]
        batch = dict(image=torch.stack([view[0] for view in views]).to(device),
                     context=torch.stack([view[1] for view in views]).to(device),
                     view_meta=[view[2] for view in views])
        logits = model(batch)
        if logits.shape != (len(views), 1, 512, 512) or not torch.isfinite(logits).all():
            raise ValueError("E4 native inference requires finite [B,1,512,512] logits")
        maps.extend(logits.sigmoid()[:, 0].cpu())
    result = stitch_tiles_hann(maps, records, (h, w), local_size=512)
    if result.shape != (h, w) or not torch.isfinite(result).all():
        raise ValueError("E4 native Hann stitching failed")
    return result
