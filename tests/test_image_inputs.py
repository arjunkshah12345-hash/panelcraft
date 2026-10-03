"""Image-input coercion.

``as_pil`` is the single place that knows how to turn whatever a caller or a provider hands
us into a PIL image. Every entry point funnels through it, so a gap here becomes a 500 on a
public endpoint rather than a local failure.

The BytesIO case is a regression test: ``/api/score`` decodes base64 data URLs into a stream
and passes it straight through, which previously raised ``TypeError: cannot interpret
<class '_io.BytesIO'> as an image``.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from api.services.benchmark import as_pil


def _png_bytes(size=(16, 16)) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((*size, 3), dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_accepts_pil_passthrough():
    image = Image.new("RGB", (8, 8))
    assert as_pil(image) is image


def test_accepts_numpy_uint8():
    assert as_pil(np.zeros((8, 8, 3), dtype=np.uint8)).size == (8, 8)


def test_promotes_float_array_to_uint8():
    """Providers may hand back floats in [0, 1]; without this every embedding is black."""
    assert as_pil(np.ones((8, 8, 3), dtype=np.float32)).mode == "RGB"


def test_expands_grayscale_to_three_channels():
    assert as_pil(np.zeros((8, 8), dtype=np.uint8)).mode == "RGB"


def test_accepts_binary_stream():
    """Regression: /api/score feeds data-URL bytes in as a BytesIO."""
    out = as_pil(io.BytesIO(_png_bytes()))
    assert out.size == (16, 16)
    assert out.mode == "RGB"


def test_accepts_path(tmp_path):
    target = tmp_path / "panel.png"
    target.write_bytes(_png_bytes())
    assert as_pil(target).size == (16, 16)


def test_rejects_unsupported_type():
    with pytest.raises(TypeError):
        as_pil(object())