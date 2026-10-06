"""Image helpers shared by backends and the ORA loop.

Loading here is centralized so we can swap to lazy streaming later without
touching every call site.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image


def load_image(path: Optional[Path | str]) -> Optional[Image.Image]:
    """Load an image from disk, returning None if path is None.

    Returns an RGB PIL.Image. Raises FileNotFoundError if the path is set but
    doesn't exist we prefer failing loudly over silently passing None.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"image not found: {p}")
    img = Image.open(p)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def image_fingerprint(img: Image.Image, size: int = 8) -> str:
    """Tiny average-hash-style fingerprint. Used by the mock backend to
    deterministically vary its outputs with the visual state so rollout
    tests actually exercise the re-encoding path.
    """
    thumb = img.convert("L").resize((size, size))
    # get_flattened_data() added in Pillow 10.4; getdata() works on all versions.
    pixels = list(
        thumb.get_flattened_data()
        if hasattr(thumb, "get_flattened_data")
        else thumb.getdata()
    )
    avg = sum(pixels) / len(pixels)
    bits = "".join("1" if p > avg else "0" for p in pixels)
    return f"{int(bits, 2):0{size * size // 4}x}"
