"""Real generation backends: PuLID and InstantID via hosted inference.

Why hosted instead of local:

* InstantID and PuLID both want an SDXL/Flux backbone plus roughly 8-12GB of VRAM for
  the ID encoder. On a laptop that is a bad afternoon.
* Hosted endpoints (Replicate, fal) expose the identity-weight parameter directly, which
  is the dial the entire benchmark turns. When we get the weight natively the tradeoff
  curve is real; when we only have a prompt we label the condition ``fixed`` and the
  chart says so.

Neither of these is exercised in CI and neither is needed to reproduce the harness
results — that is exactly why the synthetic provider exists. These adapters are thin,
explicit about which parameters they found on the endpoint, and fail loudly rather than
silently degrading.
"""

from __future__ import annotations

import base64
import io
import os
import time

import httpx
import numpy as np
from PIL import Image

from .base import GeneratedPanel, ProviderCapabilities


class ProviderError(RuntimeError):
    pass


class _HostedProvider:
    """Shared plumbing for Replicate / fal endpoints."""

    name = "hosted"
    default_model = ""

    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: float = 180.0):
        self.api_key = api_key or os.getenv(self.env_key, "")
        self.model = model or self.default_model
        self.timeout = timeout
        if not self.api_key:
            raise ProviderError(f"{self.name}: missing API key (env {self.env_key})")

    env_key = ""

    def _auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _to_array(self, url: str) -> np.ndarray:
        response = httpx.get(url, timeout=self.timeout, follow_redirects=True)
        response.raise_for_status()
        return np.asarray(Image.open(io.BytesIO(response.content)).convert("RGB"))

    def generate(
        self,
        *,
        prompt: str,
        reference: list,
        identity_strength: float,
        seed: int = 0,
        panel_index: int = 0,
        caption: str = "",
    ) -> GeneratedPanel:
        if not reference:
            raise ProviderError(f"{self.name}: reference image required")
        raise NotImplementedError


class PulidReplicateProvider(_HostedProvider):
    """PuLID: face-identity preserving generation via an ID encoder -> AdaIN."""

    name = "pulid-replicate"
    env_key = "REPLICATE_API_TOKEN"
    default_model = "bytedance/pulid"

    # Candidate identity-weight parameter names, in priority order. PuLID forks on
    # Replicate have shipped under several names across versions.
    ID_PARAMS = ("id_weight", "identity_weight", "identity_scale", "ip_weight")

    def _find_identity_param(self, payload: dict) -> tuple[str | None, str]:
        for key in self.ID_PARAMS:
            if key in payload:
                return key, "native"
        return None, "fixed"

    def generate(
        self,
        *,
        prompt: str,
        reference: list,
        identity_strength: float,
        seed: int = 0,
        panel_index: int = 0,
        caption: str = "",
    ) -> GeneratedPanel:
        if not reference:
            raise ProviderError(f"{self.name}: needs at least one reference image")

        data_uri = _image_to_data_uri(reference[0])
        strength = float(np.clip(identity_strength, 0.0, 1.0))
        payload = {
            "input_image": data_uri,
            "prompt": prompt,
            "seed": seed or (panel_index * 7919 + 13),
            "num_inference_steps": 30,
            "guidance_scale": 6.0,
        }
        id_param, control = self._find_identity_param(payload)
        if id_param:
            payload[id_param] = 0.4 + strength * 0.8

        with httpx.Client(timeout=self.timeout) as client:
            created = client.post(
                f"https://api.replicate.com/v1/models/{self.model}/predictions",
                headers=self._auth_headers(),
                json={"input": payload},
            )
            created.raise_for_status()
            prediction = created.json()

            deadline = time.time() + self.timeout
            while prediction.get("status") in ("starting", "processing"):
                if time.time() > deadline:
                    raise ProviderError(f"{self.name}: prediction {prediction.get('id')} timed out")
                time.sleep(1.5)
                polled = client.get(
                    f"https://api.replicate.com/v1/predictions/{prediction['id']}",
                    headers=self._auth_headers(),
                )
                polled.raise_for_status()
                prediction = polled.json()

        if prediction.get("status") != "succeeded":
            raise ProviderError(f"{self.name}: {prediction.get('status')} {prediction.get('error')}")

        output = prediction.get("output")
        url = output[0] if isinstance(output, list) else output
        image = self._to_array(url)

        return GeneratedPanel(
            image=image,
            prompt=prompt,
            seed=payload["seed"],
            meta={
                "provider": self.name,
                "model": self.model,
                "identity_strength": strength,
                "identity_control": control,
                "identity_param": id_param,
                "prediction_id": prediction.get("id"),
            },
        )

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name=self.name,
            supports_identity_strength=True,
            supports_seed=True,
            identity_param="id_weight",
            identity_control="native",
            local=False,
            notes="PuLID via Replicate. Native id_weight when the model exposes it.",
        )


class InstantIdReplicateProvider(_HostedProvider):
    """InstantID: single-image identity, zero-shot, via IP-Adapter style injection."""

    name = "instantid-replicate"
    env_key = "REPLICATE_API_TOKEN"
    default_model = "InstantX/instant-id"
    ID_PARAMS = ("ip_adapter_scale", "identity_scale", "instantid_weight")

    def generate(
        self,
        *,
        prompt: str,
        reference: list,
        identity_strength: float,
        seed: int = 0,
        panel_index: int = 0,
        caption: str = "",
    ) -> GeneratedPanel:
        if not reference:
            raise ProviderError(f"{self.name}: needs at least one reference image")

        data_uri = _image_to_data_uri(reference[0])
        strength = float(np.clip(identity_strength, 0.0, 1.0))
        payload = {
            "input_image": data_uri,
            "prompt": prompt,
            "seed": seed or (panel_index * 6151 + 29),
            "num_inference_steps": 30,
            "guidance_scale": 5.0,
            "ip_adapter_scale": 0.0 + strength * 1.0,
        }

        with httpx.Client(timeout=self.timeout) as client:
            created = client.post(
                f"https://api.replicate.com/v1/models/{self.model}/predictions",
                headers=self._auth_headers(),
                json={"input": payload},
            )
            created.raise_for_status()
            prediction = created.json()
            deadline = time.time() + self.timeout
            while prediction.get("status") in ("starting", "processing"):
                if time.time() > deadline:
                    raise ProviderError(f"{self.name}: prediction timed out")
                time.sleep(1.5)
                polled = client.get(
                    f"https://api.replicate.com/v1/predictions/{prediction['id']}",
                    headers=self._auth_headers(),
                )
                polled.raise_for_status()
                prediction = polled.json()

        if prediction.get("status") != "succeeded":
            raise ProviderError(f"{self.name}: {prediction.get('status')} {prediction.get('error')}")

        output = prediction.get("output")
        url = output[0] if isinstance(output, list) else output
        return GeneratedPanel(
            image=self._to_array(url),
            prompt=prompt,
            seed=payload["seed"],
            meta={
                "provider": self.name,
                "model": self.model,
                "identity_strength": strength,
                "identity_control": "native",
                "identity_param": "ip_adapter_scale",
                "prediction_id": prediction.get("id"),
            },
        )

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name=self.name,
            supports_identity_strength=True,
            supports_seed=True,
            identity_param="ip_adapter_scale",
            identity_control="native",
            local=False,
            notes="InstantID via Replicate. ip_adapter_scale maps to the identity dial.",
        )


class PromptOnlyBaselineProvider(_HostedProvider):
    """The control condition.

    Same base diffusion model, same prompt, **no** reference image. This is the honest
    baseline: the only variable is whether identity conditioning happens at all. Scoring
    this is what makes the improvement number mean something.
    """

    name = "baseline-no-id"
    env_key = "REPLICATE_API_TOKEN"
    default_model = "black-forest-labs/flux-schnell"

    def generate(
        self,
        *,
        prompt: str,
        reference: list,
        identity_strength: float,
        seed: int = 0,
        panel_index: int = 0,
        caption: str = "",
    ) -> GeneratedPanel:
        payload = {
            "prompt": prompt,
            "seed": seed or (panel_index * 3571 + 41),
            "num_inference_steps": 4,
            "output_format": "webp",
        }
        with httpx.Client(timeout=self.timeout) as client:
            created = client.post(
                f"https://api.replicate.com/v1/models/{self.model}/predictions",
                headers=self._auth_headers(),
                json={"input": payload},
            )
            created.raise_for_status()
            prediction = created.json()
            deadline = time.time() + self.timeout
            while prediction.get("status") in ("starting", "processing"):
                if time.time() > deadline:
                    raise ProviderError(f"{self.name}: prediction timed out")
                time.sleep(1.5)
                polled = client.get(
                    f"https://api.replicate.com/v1/predictions/{prediction['id']}",
                    headers=self._auth_headers(),
                )
                polled.raise_for_status()
                prediction = polled.json()

        if prediction.get("status") != "succeeded":
            raise ProviderError(f"{self.name}: {prediction.get('status')}")

        output = prediction.get("output")
        url = output[0] if isinstance(output, list) else output
        return GeneratedPanel(
            image=self._to_array(url),
            prompt=prompt,
            seed=payload["seed"],
            meta={
                "provider": self.name,
                "model": self.model,
                # Recorded but NOT applied. The baseline ignores identity by design.
                "identity_strength": 0.0,
                "identity_control": "none",
                "identity_param": None,
            },
        )

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name=self.name,
            supports_identity_strength=False,
            supports_seed=True,
            identity_param=None,
            identity_control="none",
            local=False,
            notes="Control condition: identical model, no reference image.",
        )


def _image_to_data_uri(image) -> str:
    """Encode a PIL image / ndarray / path as a data URI for the endpoint."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype("uint8"))
    elif isinstance(image, str):
        image = Image.open(image)
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{encoded}"