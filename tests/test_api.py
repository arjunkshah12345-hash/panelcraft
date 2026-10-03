"""HTTP surface.

These exercise the routes a judge would actually hit, including the two defects found in
review:

* ``POST /api/score`` decoded base64 data URLs into a ``BytesIO`` and handed it to a
  normaliser that only understood PIL/ndarray/path, so every data-URL request 400'd.
* ``POST /api/generate`` returned panels without scoring them, which quietly shipped the
  weaker version of the product.

Generation and scoring load FaceNet, so this module costs one model load per session
rather than one per test.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _data_uri(size=(96, 96), colour=(120, 90, 70)) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.full((*size, 3), colour, dtype=np.uint8)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


# ------------------------------------------------------------------ cheap routes


def test_health(client):
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert "synthetic" in body["providers"]


def test_capabilities_advertises_synthetic_without_a_token(client):
    """A fresh deploy must be demoable before anyone sets REPLICATE_API_TOKEN."""
    catalogue = client.get("/api/capabilities").json()
    synthetic = next(p for p in catalogue if p["name"] == "synthetic")
    assert synthetic["measurement_domain"] == "synthetic"
    assert synthetic["provides_ground_truth"] is True


def test_default_script_has_panels(client):
    body = client.get("/api/default-script").json()
    assert body["panels"]
    assert body["character"]["name"]


# ------------------------------------------------------------------ score endpoint


def test_score_accepts_data_urls(client):
    """Regression: data URLs raised TypeError -> 400 on every request."""
    response = client.post(
        "/api/score",
        json={
            "reference_images": [_data_uri(), _data_uri(colour=(130, 100, 80))],
            "panel_images": [_data_uri(), _data_uri(colour=(110, 85, 65))],
            "panel_ids": ["p0", "p1"],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["identity_mean"] is not None
    assert [p["panel_id"] for p in body["panels"]] == ["p0", "p1"]
    assert "consistent" in body


def test_score_verdict_follows_threshold(client):
    """A threshold the caller supplies must reach the per-panel verdicts."""
    images = {"reference_images": [_data_uri()], "panel_images": [_data_uri()]}
    lenient = client.post("/api/score", json={**images, "threshold": -1.0}).json()
    strict = client.post("/api/score", json={**images, "threshold": 1.0}).json()
    assert lenient["panels"][0]["verdict"] == "consistent"
    assert strict["panels"][0]["verdict"] == "drift"


def test_score_rejects_empty_input(client):
    response = client.post(
        "/api/score", json={"reference_images": [], "panel_images": [_data_uri()]}
    )
    assert response.status_code == 400


def test_score_rejects_unreadable_image(client):
    response = client.post(
        "/api/score",
        json={"reference_images": [_data_uri()], "panel_images": ["data:image/png;base64,%%%"]},
    )
    assert response.status_code == 400


def test_score_rejects_out_of_range_threshold(client):
    response = client.post(
        "/api/score",
        json={
            "reference_images": [_data_uri()],
            "panel_images": [_data_uri()],
            "threshold": 5.0,
        },
    )
    assert response.status_code == 422


# ------------------------------------------------------------------ generate endpoint


@pytest.fixture(scope="module")
def script(client):
    return client.get("/api/default-script").json()


def test_generate_scores_its_own_panels(client, script):
    """The comic must arrive already measured — that is the product, not a nice extra."""
    body = client.post(
        "/api/generate",
        json={"script": script, "provider": "synthetic", "identity_strength": 0.5},
    ).json()
    assert len(body["panels"]) == len(script["panels"])
    assert body["report"]["identity_mean"] is not None
    for panel in body["panels"]:
        assert panel["image_url"].startswith("/api/images/")


def test_generated_image_is_retrievable(client, script):
    url = client.post(
        "/api/generate",
        json={"script": script, "provider": "synthetic", "identity_strength": 0.5},
    ).json()["panels"][0]["image_url"]
    assert client.get(url).status_code == 200


def test_generate_requires_a_script(client):
    assert client.post("/api/generate", json={"provider": "synthetic"}).status_code == 422