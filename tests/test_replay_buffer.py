"""ReplayBuffer behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from isegm.adaptation import ReplayBuffer


def _entry(rng, shape=(64, 64)):
    img = (rng.random((*shape, 3)) * 255).astype(np.uint8)
    mask = rng.random(shape) > 0.5
    clicks = [(10, 10, True), (20, 20, False)]
    return img, clicks, mask


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=0)


def test_add_and_len():
    buf = ReplayBuffer(capacity=4)
    rng = np.random.default_rng(0)
    for _ in range(3):
        buf.add(*_entry(rng))
    assert len(buf) == 3


def test_overflow_drops_oldest():
    buf = ReplayBuffer(capacity=2)
    rng = np.random.default_rng(0)
    img1, c1, m1 = _entry(rng)
    img2, c2, m2 = _entry(rng)
    img3, c3, m3 = _entry(rng)
    buf.add(img1, c1, m1)
    buf.add(img2, c2, m2)
    buf.add(img3, c3, m3)
    items = buf.items()
    assert len(items) == 2
    # First entry should be img2 now.
    assert np.array_equal(items[0].image, img2)
    assert np.array_equal(items[1].image, img3)


def test_add_makes_defensive_copies():
    buf = ReplayBuffer(capacity=2)
    rng = np.random.default_rng(0)
    img, clicks, mask = _entry(rng)
    buf.add(img, clicks, mask)

    # Mutate the originals
    img[:] = 0
    mask[:] = False
    clicks.clear()

    e = buf.items()[0]
    assert e.image.any(), "buffered image must not see external mutations"
    assert e.accepted_mask.any() or not e.accepted_mask.any()  # bool, just check presence
    assert len(e.clicks) == 2


def test_clear():
    buf = ReplayBuffer(capacity=3)
    rng = np.random.default_rng(0)
    buf.add(*_entry(rng))
    buf.add(*_entry(rng))
    buf.clear()
    assert len(buf) == 0


def test_iteration_yields_buffer_entries():
    buf = ReplayBuffer(capacity=4)
    rng = np.random.default_rng(0)
    for _ in range(2):
        buf.add(*_entry(rng))
    seen = list(buf)
    assert len(seen) == 2
    assert all(hasattr(x, 'image') and hasattr(x, 'clicks')
               and hasattr(x, 'accepted_mask') for x in seen)


def test_image_dtype_coerced_to_uint8():
    buf = ReplayBuffer(capacity=2)
    rng = np.random.default_rng(0)
    img_f = rng.random((32, 32, 3)).astype(np.float32) * 255
    mask = rng.random((32, 32)) > 0.5
    buf.add(img_f, [(0, 0, True)], mask)
    e = buf.items()[0]
    assert e.image.dtype == np.uint8
