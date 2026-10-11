"""Class-balanced Bernoulli consistency, with a detached clean-image teacher.

The supervised clean BCE + Dice is essential: consistency alone admits a
constant all-background solution. This loss is not a brightness reconstruction
objective and never makes the anomaly mask a target for lighting itself.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .anomaly_loss import _validate_binary_segmentation_inputs


class IlluminationConsistencyLoss(nn.Module):
    def forward(self, illuminated_logits, clean_logits, target):
        _validate_binary_segmentation_inputs(illuminated_logits, target)
        _validate_binary_segmentation_inputs(clean_logits, target)
        dtype = torch.float64 if torch.float64 in (clean_logits.dtype, illuminated_logits.dtype) else torch.float32
        student = illuminated_logits.to(dtype)
        teacher = clean_logits.detach().to(dtype)
        probability = teacher.sigmoid()
        # KL(Bernoulli(p_clean) || Bernoulli(p_illuminated)). Stable even if
        # the student saturates in the wrong direction; no log(sigmoid(x)).
        pixel = (F.binary_cross_entropy_with_logits(student, probability, reduction='none')
                 - F.binary_cross_entropy_with_logits(teacher, probability, reduction='none')).clamp_min(0)
        y, errors = target.to(dtype).flatten(1), pixel.flatten(1)
        positives, negatives = y.sum(1), (1-y).sum(1)
        fg = (errors*y).sum(1) / positives.clamp_min(1)
        bg = (errors*(1-y)).sum(1) / negatives.clamp_min(1)
        classes = (positives > 0).to(dtype) + (negatives > 0).to(dtype)
        return ((fg+bg) / classes.clamp_min(1)).mean()
