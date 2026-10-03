"""Provider interface.

Everything downstream — scoring, the benchmark sweep, the API, the UI — talks only to
this. Adding a real generation backend must never require touching the harness.

The one contract that matters: ``identity_strength`` in [0, 1], where 0 means "no
identity information reaches the model" and 1 means "identity is fully pinned".
Implementations that cannot modulate the dial must say so in
``ProviderCapabilities.identity_control`` rather than silently ignoring it, because the
whole tradeoff curve is meaningless if the parameter is a no-op.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class GeneratedPanel:
    """One generated panel plus the metadata needed to reproduce it."""

    image: np.ndarray  # RGB uint8 HxWx3
    prompt: str
    seed: int
    meta: dict = field(default_factory=dict)
    # (x1, y1, x2, y2) in pixel coords. Generators that know their own composition
    # supply it, which spares us a detection step that may fail on synthetic art.
    face_box: tuple[int, int, int, int] | None = None


@dataclass
class ProviderCapabilities:
    name: str
    supports_identity_strength: bool = True
    supports_seed: bool = True
    identity_param: str | None = "identity_strength"
    identity_control: str = "native"  # native | prompt-only | fixed | none
    local: bool = False
    provides_face_box: bool = False
    # "photographic" faces can be measured by a photo-trained face embedding model.
    # "synthetic" faces cannot, and a benchmark must say so rather than report a number.
    measurement_domain: str = "photographic"
    # True when the provider knows the true identity drift, enabling harness validation
    # against ground truth instead of trusting the measurement blindly.
    provides_ground_truth: bool = False
    # The underlying diffusion backbone, e.g. "FLUX.1-dev" or "SDXL". Recorded because a
    # prompt-only control is only a *controlled* ablation if it shares this with the
    # treatment — otherwise the comparison confounds identity with the base model.
    base_model: str | None = None
    notes: str = ""


@runtime_checkable
class Provider(Protocol):
    name: str

    def generate(
        self,
        *,
        prompt: str,
        reference: list,
        identity_strength: float,
        seed: int = 0,
        panel_index: int = 0,
        caption: str = "",
    ) -> GeneratedPanel: ...

    def capabilities(self) -> ProviderCapabilities: ...