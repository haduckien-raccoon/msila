"""MS-ILA Day-05 representation selector.

Project-relative path:
    msila/src/models/feature_selector.py

The module performs exactly one scientific operation: choose which already
prepared DINOv3 feature sources are visible to the common downstream pipeline.
It does NOT align Context, project channels, adapt features, fuse features, or
run the decoder.

Expected canonical feature keys:
    local_b4, local_b8, local_b12,
    context_b4, context_b8, context_b12

Day-05 candidates:
    R0 / deep_only           -> local_b12
    R1 / multi_local         -> local_b4, local_b8, local_b12
    R2 / multi_local_context -> all six Local+Context sources

For R2, Context->Local geometric alignment and channel projection must already
have happened upstream. Therefore, when ``validate=True``, all selected tensors
are required to have identical BCHW shapes so the same Mean Fusion can be used
without introducing another confound.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final, Literal

from torch import Tensor, nn


RepresentationMode = Literal[
    "deep_only",
    "multi_local",
    "multi_local_context",
]


MODE_TO_KEYS: Final[dict[str, tuple[str, ...]]] = {
    "deep_only": (
        "local_b12",
    ),
    "multi_local": (
        "local_b4",
        "local_b8",
        "local_b12",
    ),
    "multi_local_context": (
        "local_b4",
        "local_b8",
        "local_b12",
        "context_b4",
        "context_b8",
        "context_b12",
    ),
}

CANDIDATE_TO_MODE: Final[dict[str, RepresentationMode]] = {
    "R0": "deep_only",
    "R1": "multi_local",
    "R2": "multi_local_context",
}


class FeatureSelectionError(ValueError):
    """Raised when the Day-05 representation contract is violated."""


class FeatureSelector(nn.Module):
    """Select one of the three locked Day-05 representations.

    Parameters
    ----------
    mode:
        One of ``deep_only``, ``multi_local``, ``multi_local_context``.
        ``R0``, ``R1`` and ``R2`` are accepted as convenience aliases.

    validate:
        If True, validate the selected tensors as BCHW and require them to be
        compatible with the shared downstream Mean Fusion.

    Returns
    -------
    list[Tensor]
        Selected tensors in deterministic scientific order.
    """

    def __init__(
        self,
        mode: RepresentationMode | str,
        *,
        validate: bool = True,
    ) -> None:
        super().__init__()

        normalized_mode = CANDIDATE_TO_MODE.get(str(mode).upper(), str(mode))
        if normalized_mode not in MODE_TO_KEYS:
            allowed = sorted((*MODE_TO_KEYS.keys(), *CANDIDATE_TO_MODE.keys()))
            raise FeatureSelectionError(
                f"Unknown representation mode {mode!r}. Allowed: {allowed}"
            )

        self.mode: RepresentationMode = normalized_mode  # type: ignore[assignment]
        self.validate = bool(validate)

    @property
    def source_keys(self) -> tuple[str, ...]:
        """Canonical feature keys used by the current representation."""

        return MODE_TO_KEYS[self.mode]

    @property
    def num_sources(self) -> int:
        """Number of selected feature sources: 1, 3, or 6."""

        return len(self.source_keys)

    @classmethod
    def from_candidate(
        cls,
        candidate_id: str,
        *,
        validate: bool = True,
    ) -> "FeatureSelector":
        """Construct from the experiment IDs R0/R1/R2."""

        candidate_id = str(candidate_id).upper()
        if candidate_id not in CANDIDATE_TO_MODE:
            raise FeatureSelectionError(
                f"Unknown candidate {candidate_id!r}; expected one of "
                f"{tuple(CANDIDATE_TO_MODE)}."
            )
        return cls(CANDIDATE_TO_MODE[candidate_id], validate=validate)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
        candidate_id: str,
        *,
        validate: bool = True,
    ) -> "FeatureSelector":
        """Build directly from ``configs/day05_representation.yaml``.

        This helper intentionally reads only:
            representations.<candidate_id>.mode

        It does not merge or modify Adapter/Fusion/Decoder/Loss settings.
        """

        candidate_id = str(candidate_id).upper()
        representations = config.get("representations")
        if not isinstance(representations, Mapping):
            raise FeatureSelectionError(
                "Config must contain a 'representations' mapping."
            )

        candidate = representations.get(candidate_id)
        if not isinstance(candidate, Mapping):
            raise FeatureSelectionError(
                f"Config has no representation candidate {candidate_id!r}."
            )

        mode = candidate.get("mode")
        if not isinstance(mode, str):
            raise FeatureSelectionError(
                f"representations.{candidate_id}.mode must be a string."
            )

        selector = cls(mode, validate=validate)

        # Optional audit: if YAML also lists `sources`, it must exactly match
        # the code contract. This catches config/code drift immediately.
        configured_sources = candidate.get("sources")
        if configured_sources is not None:
            if not isinstance(configured_sources, Sequence) or isinstance(
                configured_sources, (str, bytes)
            ):
                raise FeatureSelectionError(
                    f"representations.{candidate_id}.sources must be a sequence."
                )
            configured_sources = tuple(str(x) for x in configured_sources)
            if configured_sources != selector.source_keys:
                raise FeatureSelectionError(
                    f"Config/code drift for {candidate_id}: "
                    f"config sources={configured_sources}, "
                    f"expected={selector.source_keys}."
                )

        return selector

    def forward(self, features: Mapping[str, Tensor]) -> list[Tensor]:
        """Return the selected tensors in deterministic source order."""

        if not isinstance(features, Mapping):
            raise TypeError(
                "FeatureSelector expects Mapping[str, Tensor], "
                f"got {type(features)!r}."
            )

        missing = [key for key in self.source_keys if key not in features]
        if missing:
            raise FeatureSelectionError(
                f"Missing feature source(s) for mode={self.mode}: {missing}. "
                f"Available keys: {sorted(map(str, features.keys()))}"
            )

        selected = [features[key] for key in self.source_keys]

        if self.validate:
            self._validate_selected(selected)

        return selected

    def _validate_selected(self, selected: Sequence[Tensor]) -> None:
        if len(selected) != self.num_sources:
            raise FeatureSelectionError(
                f"Expected {self.num_sources} sources for {self.mode}, "
                f"got {len(selected)}."
            )

        reference: Tensor | None = None
        reference_key: str | None = None

        for key, tensor in zip(self.source_keys, selected):
            if not isinstance(tensor, Tensor):
                raise TypeError(
                    f"Feature {key!r} must be torch.Tensor, "
                    f"got {type(tensor)!r}."
                )
            if tensor.ndim != 4:
                raise FeatureSelectionError(
                    f"Feature {key!r} must be BCHW [B,C,H,W], "
                    f"got shape={tuple(tensor.shape)}."
                )

            if reference is None:
                reference = tensor
                reference_key = key
                continue

            # Mean fusion requires exact shape compatibility. For R2 this is
            # also a direct guard that Context->Local alignment/projection has
            # already been completed upstream.
            if tensor.shape != reference.shape:
                raise FeatureSelectionError(
                    f"Selected features must have identical shapes before "
                    f"shared Mean Fusion: {reference_key}={tuple(reference.shape)} "
                    f"but {key}={tuple(tensor.shape)}. "
                    "For Context features, run Context->Local alignment and "
                    "the locked projection before FeatureSelector."
                )

            if tensor.dtype != reference.dtype:
                raise FeatureSelectionError(
                    f"Selected feature dtype mismatch: {reference_key}="
                    f"{reference.dtype}, {key}={tensor.dtype}."
                )

            if tensor.device != reference.device:
                raise FeatureSelectionError(
                    f"Selected feature device mismatch: {reference_key}="
                    f"{reference.device}, {key}={tensor.device}."
                )

    def extra_repr(self) -> str:
        return (
            f"mode={self.mode!r}, "
            f"num_sources={self.num_sources}, "
            f"sources={self.source_keys}, "
            f"validate={self.validate}"
        )
