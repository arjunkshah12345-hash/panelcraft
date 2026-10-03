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
    # Declared per subclass. These are *model inputs*, so there is nothing to discover by
    # probing a response — an earlier version scanned the request payload it had just built,
    # never found the key, and silently dropped every strength value while still advertising
    # the dial as native, making the benchmark's x-axis a no-op. Declaring the parameter and
    # its native range keeps the mapping auditable instead of implied.
    identity_param: str | None = None
    identity_range: tuple[float, float] = (0.0, 1.0)

    def map_strength(self, identity_strength: float) -> float:
        """Map the normalised 0-1 dial onto this model's native identity-weight range."""
        low, high = self.identity_range
        strength = float(np.clip(identity_strength, 0.0, 1.0))
        return low + strength * (high - low)

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

    identity_param = "id_weight"
    # PuLID loses the face below ~0.3 and starts to distort past ~1.3, so the dial spans
    # that usable window rather than a naive 0-1.
    identity_range = (0.4, 1.2)

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
        native_weight = self.map_strength(identity_strength)
        payload = {
            "input_image": data_uri,
            "prompt": prompt,
            "seed": seed or (panel_index * 7919 + 13),
            "num_inference_steps": 30,
            "guidance_scale": 6.0,
            self.identity_param: native_weight,
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
                "identity_control": "native",
                "identity_param": self.identity_param,
                "identity_weight_applied": native_weight,
                "prediction_id": prediction.get("id"),
            },
        )

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name=self.name,
            supports_identity_strength=True,
            supports_seed=True,
            identity_param=self.identity_param,
            identity_control="native",
            local=False,
            base_model="FLUX.1-dev",
            notes=(
                f"PuLID via Replicate. id_weight maps the 0-1 dial onto "
                f"{self.identity_range[0]}-{self.identity_range[1]}."
            ),
        )


class InstantIdReplicateProvider(_HostedProvider):
    """InstantID: single-image identity, zero-shot, via IP-Adapter style injection."""

    name = "instantid-replicate"
    env_key = "REPLICATE_API_TOKEN"
    default_model = "InstantX/instant-id"
    identity_param = "ip_adapter_scale"
    identity_range = (0.0, 1.0)

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
        native_weight = self.map_strength(identity_strength)
        payload = {
            "input_image": data_uri,
            "prompt": prompt,
            "seed": seed or (panel_index * 6151 + 29),
            "num_inference_steps": 30,
            "guidance_scale": 5.0,
            self.identity_param: native_weight,
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
                "identity_param": self.identity_param,
                "identity_weight_applied": native_weight,
                "prediction_id": prediction.get("id"),
            },
        )

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name=self.name,
            supports_identity_strength=True,
            supports_seed=True,
            identity_param=self.identity_param,
            identity_control="native",
            local=False,
            base_model="SDXL",
            notes=(
                f"InstantID via Replicate. {self.identity_param} maps the 0-1 dial onto "
                f"{self.identity_range[0]}-{self.identity_range[1]}."
            ),
        )


class PromptOnlyBaselineProvider(_HostedProvider):
    """The control condition: same prompt, no reference image, no identity conditioning.

    Honest caveat, because it matters for how the result may be read: the default model is
    ``flux-schnell``, while PuLID runs on a FLUX.1-dev-class backbone and InstantID on
    SDXL. This is therefore **not** a controlled ablation — it differs from the treatment
    in two variables at once (identity conditioning *and* backbone), so a gap against it
    cannot be attributed to identity alone. It answers "how far does prompt-only drift on
    a comparable model?" not "how much did the ID encoder buy?"

    To make it a genuine control, point this at the same backbone as the treatment via
    ``model=``; ``base_model`` is recorded on the capabilities so the pairing is visible
    in the UI instead of assumed by whoever reads the chart.
    """

    name = "baseline-no-id"
    env_key = "REPLICATE_API_TOKEN"
    default_model = "black-forest-labs/flux-schnell"
    base_model = "FLUX.1-schnell"

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
            base_model=self.base_model,
            notes=(
                "Prompt-only reference point. NOT a controlled ablation against the "
                f"treatment unless the backbone is matched ({self.base_model} vs the "
                "treatment's own backbone)."
            ),
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