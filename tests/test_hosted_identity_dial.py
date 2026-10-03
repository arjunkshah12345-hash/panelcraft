"""The identity dial must actually reach the model.

This is the highest-value regression in the repo. The benchmark's entire x-axis is
"identity strength", so a provider that accepts the dial and quietly drops it does not
degrade the chart — it silently mismeasures it, and reports a confident curve about a knob
it never turned.

The original hosted adapter looked for its identity parameter inside the request payload it
had just constructed, found nothing, marked the control ``fixed``, and never applied it. The
capabilities endpoint still advertised ``native``, so the UI showed a working dial. These
tests intercept the HTTP call and assert on the bytes that go over the wire.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from api.services.providers import hosted
from api.services.providers.hosted import (
    InstantIdReplicateProvider,
    PromptOnlyBaselineProvider,
    ProviderError,
    PulidReplicateProvider,
)


def _reference() -> Image.Image:
    return Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))


def _png_response() -> object:
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(buffer, format="PNG")

    class Response:
        content = buffer.getvalue()

        def raise_for_status(self):
            return None

    return Response()


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    """Captures the prediction input instead of calling Replicate."""

    captured: dict = {}

    def __init__(self, *args, **kwargs):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, headers=None, json=None, **kwargs):
        type(self).captured["url"] = url
        type(self).captured["input"] = (json or {}).get("input", {})
        return _FakeResponse(
            {"id": "pred-1", "status": "succeeded", "output": "https://example.test/i.png"}
        )

    def get(self, url, headers=None, **kwargs):
        return _FakeResponse(
            {"id": "pred-1", "status": "succeeded", "output": "https://example.test/i.png"}
        )


@pytest.fixture
def captured(monkeypatch):
    _FakeClient.captured = {}
    monkeypatch.setattr(hosted.httpx, "Client", _FakeClient)
    monkeypatch.setattr(hosted.httpx, "get", lambda *a, **k: _png_response())
    return _FakeClient.captured


@pytest.mark.parametrize(
    "provider_cls, param",
    [
        (PulidReplicateProvider, "id_weight"),
        (InstantIdReplicateProvider, "ip_adapter_scale"),
    ],
)
def test_identity_param_is_present_in_request(captured, provider_cls, param):
    provider = provider_cls(api_key="test")
    provider.generate(
        prompt="a lighthouse keeper",
        reference=[_reference()],
        identity_strength=0.75,
        seed=42,
    )
    assert param in captured["input"], f"{provider.name} dropped {param} on the wire"


@pytest.mark.parametrize(
    "provider_cls, param, low, high",
    [
        (PulidReplicateProvider, "id_weight", 0.4, 1.2),
        (InstantIdReplicateProvider, "ip_adapter_scale", 0.0, 1.0),
    ],
)
def test_strength_maps_to_declared_native_range(provider_cls, param, low, high):
    provider = provider_cls(api_key="test")
    assert provider.map_strength(0.0) == pytest.approx(low)
    assert provider.map_strength(1.0) == pytest.approx(high)
    assert provider.map_strength(0.5) == pytest.approx((low + high) / 2)


def test_mapping_is_monotonic_and_clamped():
    """A non-monotonic dial would make the tradeoff curve meaningless."""
    provider = PulidReplicateProvider(api_key="test")
    values = [provider.map_strength(i / 10) for i in range(11)]
    assert values == sorted(values)
    assert provider.map_strength(-5) == pytest.approx(0.4)
    assert provider.map_strength(5) == pytest.approx(1.2)


def test_generated_metadata_records_applied_weight(captured):
    provider = PulidReplicateProvider(api_key="test")
    panel = provider.generate(
        prompt="p", reference=[_reference()], identity_strength=1.0, seed=1
    )
    assert panel.meta["identity_weight_applied"] == pytest.approx(1.2)
    assert panel.meta["identity_control"] == "native"


def test_capabilities_agree_with_behaviour():
    """Declared capabilities must match what generate() actually does."""
    for provider in (
        PulidReplicateProvider(api_key="test"),
        InstantIdReplicateProvider(api_key="test"),
    ):
        caps = provider.capabilities()
        assert caps.identity_control == "native"
        assert caps.identity_param == provider.identity_param
        assert caps.base_model, "record the backbone so control comparisons stay auditable"


def test_baseline_declares_no_identity_control(captured):
    provider = PromptOnlyBaselineProvider(api_key="test")
    caps = provider.capabilities()
    assert caps.identity_control == "none"
    assert caps.supports_identity_strength is False
    # The baseline must never send an identity parameter, whatever strength it is handed.
    provider.generate(prompt="p", reference=[], identity_strength=1.0, seed=1)
    assert "id_weight" not in captured["input"]
    assert "ip_adapter_scale" not in captured["input"]


def test_missing_api_key_fails_loudly():
    """Silently falling back would look like a successful run with fake numbers."""
    with pytest.raises(ProviderError):
        PulidReplicateProvider(api_key="")


def test_missing_reference_fails_loudly():
    with pytest.raises(ProviderError):
        PulidReplicateProvider(api_key="test").generate(
            prompt="p", reference=[], identity_strength=0.5
        )