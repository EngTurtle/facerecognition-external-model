"""Thread-safe lazy loading of the InsightFace engine."""

import logging
import threading

import numpy as np
from insightface.app import FaceAnalysis

from facerec.config import Config, MODEL_ROOT
from facerec.providers import select_providers

log = logging.getLogger(__name__)


class FaceModel:
    """Thread-safe lazy wrapper around insightface.app.FaceAnalysis."""

    def __init__(self, config: Config) -> None:
        """Store config; the underlying engine is created on first use."""
        self._config = config
        self._engine: FaceAnalysis | None = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        """Whether the underlying engine has been constructed."""
        return self._engine is not None

    def load(self) -> None:
        """Idempotently construct and prepare the InsightFace engine."""
        if self._engine is not None:
            return

        with self._lock:
            if self._engine is not None:
                return

            config = self._config
            log.info("Loading InsightFace model: %s", config.model_name)
            log.info("Using device: %s", config.device)

            providers = select_providers(config)
            log.info("Execution providers: %s", providers)
            log.info("Using embedding scale factor: %s", config.embedding_scale)

            engine = FaceAnalysis(
                name=config.model_name,
                root=MODEL_ROOT,
                allowed_modules=["detection", "recognition"],
                providers=providers,
            )

            ctx_id = 0 if config.device in ("cuda", "openvino") else -1
            engine.prepare(ctx_id=ctx_id, det_size=config.det_size)

            self._engine = engine
            log.info("InsightFace model loaded successfully")

    def get_faces(self, img: np.ndarray) -> list:
        """Load the engine if needed and return detected faces for img."""
        self.load()
        return self._engine.get(img)
