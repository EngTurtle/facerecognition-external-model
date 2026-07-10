"""Image decoding helpers (numpy + cv2)."""

import cv2
import numpy as np


def decode_image(data: bytes) -> np.ndarray:
    """Decode image bytes into an RGB numpy array.

    Decodes entirely in memory (no temp file, no filename collisions).
    Converts BGR to RGB to match the original service's behavior; changing
    this would change the resulting embeddings.
    """
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
