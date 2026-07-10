"""Runtime configuration: constants, env parsing, and the Config dataclass.

Stdlib only — this module must be importable without any third-party
dependency installed.
"""

import logging
import os
from dataclasses import dataclass
from typing import Mapping

log = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "buffalo_l"
DEFAULT_DET_SIZE = "640,640"
DEFAULT_EMBEDDING_SCALE = 0.017
DEFAULT_API_KEY = "some-super-secret-api-key"
API_KEY_FILE = "api.key"
MODEL_ROOT = "./models"
# Advertised via /open so Nextcloud's client-side size check always passes.
MAXIMUM_AREA = 5_000_000


def parse_det_size(value: str) -> tuple[int, int]:
    """Parse a "width,height" or single-integer det size string.

    Each dimension is snapped down to a multiple of 32 with a floor of 32.
    Falls back to the parsed default detection size on any parse error.
    """
    try:
        if "," in value:
            width_str, height_str = value.split(",", 1)
            width = int(width_str.strip())
            height = int(height_str.strip())
        else:
            size = int(value.strip())
            width, height = size, size

        width = max((width // 32) * 32, 32)
        height = max((height // 32) * 32, 32)

        return (width, height)
    except (ValueError, AttributeError):
        return parse_det_size(DEFAULT_DET_SIZE)


def load_api_key(environ: Mapping[str, str] = os.environ) -> str:
    """Resolve the API key: env var, then api.key file, then insecure default."""
    if "API_KEY" in environ:
        return environ["API_KEY"]

    try:
        with open(API_KEY_FILE, "r") as apikey:
            return apikey.read().replace("\n", "")
    except FileNotFoundError:
        log.warning("No API_KEY env var or %s file found; using insecure default API key", API_KEY_FILE)
        return DEFAULT_API_KEY


@dataclass(frozen=True)
class Config:
    """Resolved service configuration."""

    model_name: str
    device: str
    det_size: tuple[int, int]
    embedding_scale: float
    api_key: str
    cuda_device_id: str
    openvino_device_type: str
    openvino_precision: str

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> "Config":
        """Build a Config by reading the process environment."""
        return cls(
            model_name=environ.get("MODEL_NAME", DEFAULT_MODEL_NAME),
            device=environ.get("DEVICE", "cpu").lower(),
            det_size=parse_det_size(environ.get("MAX_DET_SIZE", DEFAULT_DET_SIZE)),
            embedding_scale=float(environ.get("EMBEDDING_SCALE", str(DEFAULT_EMBEDDING_SCALE))),
            api_key=load_api_key(environ),
            cuda_device_id=environ.get("CUDA_DEVICE_ID", "0"),
            openvino_device_type=environ.get("OPENVINO_DEVICE_TYPE", "GPU"),
            openvino_precision=environ.get("OPENVINO_PREC", "FP32"),
        )
