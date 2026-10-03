"""Provider registry and selection.

``PANELCRAFT_PROVIDER`` picks the active backend:

* ``synthetic`` (default) — local, deterministic, zero cost. The harness reference.
* ``pulid`` / ``instantid`` / ``baseline`` — hosted, require REPLICATE_API_TOKEN.

Hosted providers are constructed lazily so that a missing key produces a clear error at
call time instead of an import-time crash that would take the whole API down.
"""

from __future__ import annotations

import os
from dataclasses import asdict

from .base import GeneratedPanel, Provider, ProviderCapabilities
from .synthetic import SyntheticProvider

_REGISTRY: dict[str, str] = {
    "synthetic": "SyntheticProvider",
    "pulid": "PulidReplicateProvider",
    "instantid": "InstantIdReplicateProvider",
    "baseline": "PromptOnlyBaselineProvider",
}

_instances: dict[str, Provider] = {}


def available_providers() -> list[dict]:
    """Catalogue for the UI dropdown, with availability resolved."""
    from .base import ProviderCapabilities as _C  # noqa: F401  (keeps import shape obvious)

    catalogue = []
    for key in _REGISTRY:
        caps = describe(key)
        catalogue.append(caps.to_dict() if hasattr(caps, "to_dict") else _cap_dict(caps))
    return catalogue


def describe(name: str):
    try:
        return get_provider(name).capabilities()
    except Exception as exc:
        from .base import ProviderCapabilities

        return ProviderCapabilities(
            name=name,
            supports_identity_strength=False,
            identity_control="unavailable",
            notes=str(exc),
        )


def _cap_dict(caps) -> dict:
    """Serialise the whole dataclass.

    Written out by hand once, this dropped ``provides_ground_truth`` — the UI had no way to
    tell which providers could be checked against ground truth, because the field was
    present on the object and absent from the API. Deriving from the dataclass means a new
    field cannot go missing here again.
    """
    return asdict(caps)


def get_provider(name: str | None = None) -> Provider:
    """Return a cached provider instance."""
    name = (name or os.getenv("PANELCRAFT_PROVIDER") or "synthetic").lower()
    if name in _instances:
        return _instances[name]
    if name not in _REGISTRY:
        raise KeyError(f"unknown provider {name!r}; known: {sorted(_REGISTRY)}")

    if name == "synthetic":
        provider = SyntheticProvider()
    else:
        from . import hosted

        provider = {
            "pulid": hosted.PulidReplicateProvider,
            "instantid": hosted.InstantIdReplicateProvider,
            "baseline": hosted.PromptOnlyBaselineProvider,
        }[name]()

    _instances[name] = provider
    return provider


__all__ = [
    "GeneratedPanel",
    "Provider",
    "ProviderCapabilities",
    "SyntheticProvider",
    "available_providers",
    "describe",
    "get_provider",
]