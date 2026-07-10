"""Flask blueprint exposing the detect/compute/open/health/welcome routes."""

import json
import logging
import os

from flask import Blueprint, abort, current_app, request

from facerec import PACKAGE_VERSION
from facerec.auth import require_api_key
from facerec.config import Config, MAXIMUM_AREA
from facerec.faces import best_face_match, serialize_face, serialize_landmarks
from facerec.images import decode_image
from facerec.model import FaceModel

log = logging.getLogger(__name__)

bp = Blueprint("api", __name__)


def _model() -> FaceModel:
    """Return the FaceModel registered on the current app."""
    return current_app.extensions["face_model"]


def _config() -> Config:
    """Return the Config registered on the current app."""
    return current_app.extensions["facerec_config"]


def parse_min_score(form, default: float = 0.5) -> float:
    """Parse minScore from either a top-level form field or a JSON `options` field."""
    try:
        if "minScore" in form:
            return float(form.get("minScore", default))
        if "options" in form:
            options = json.loads(form.get("options", "{}"))
            return float(options.get("minScore", default))
    except (ValueError, json.JSONDecodeError):
        pass
    return default


@bp.route("/detect", methods=["POST"])
@require_api_key
def detect_faces() -> dict:
    """Detect faces in an uploaded image."""
    uploaded_file = request.files.get("file")
    if not uploaded_file:
        abort(400, "No file provided")

    filename = os.path.basename(uploaded_file.filename or "")

    try:
        img = decode_image(uploaded_file.read())
    except ValueError:
        abort(400, "Invalid or unreadable image")

    min_score = parse_min_score(request.form)

    faces = _model().get_faces(img)
    faces = [face for face in faces if face.det_score >= min_score]
    serialized_faces = [serialize_face(face, _config().embedding_scale) for face in faces]

    return {
        "filename": filename,
        "faces-count": len(serialized_faces),
        "faces": serialized_faces,
    }


@bp.route("/compute", methods=["POST"])
@require_api_key
def compute() -> dict:
    """Compute the embedding for the face closest to a given bounding box."""
    uploaded_file = request.files.get("file")
    face_json_str = request.form.get("face")

    if not uploaded_file or not face_json_str:
        abort(400, "Missing file or face data")

    try:
        face_data = json.loads(face_json_str)
    except json.JSONDecodeError:
        abort(400, "Invalid face JSON")

    filename = os.path.basename(uploaded_file.filename or "")

    try:
        img = decode_image(uploaded_file.read())
    except ValueError:
        abort(400, "Invalid or unreadable image")

    faces = _model().get_faces(img)

    target_box = [
        face_data.get("left", 0),
        face_data.get("top", 0),
        face_data.get("right", 0),
        face_data.get("bottom", 0),
    ]

    best_match = best_face_match(faces, target_box)

    if best_match is None:
        face_data["descriptor"] = []
        face_data["landmarks"] = []
        return {"filename": filename, "face": face_data}

    if best_match.kps is not None:
        face_data["landmarks"] = serialize_landmarks(best_match.kps)

    embedding = best_match.embedding * _config().embedding_scale
    face_data["descriptor"] = embedding.tolist()

    return {"filename": filename, "face": face_data}


@bp.route("/open")
@require_api_key
def open_model() -> dict:
    """Pre-load the model and advertise service configuration."""
    _model().load()
    return {
        "preferred_mimetype": "image/jpeg",
        "maximum_area": MAXIMUM_AREA,
        "model": _config().model_name,
    }


@bp.route("/health")
def health():
    """Liveness check reflecting whether the model has been loaded."""
    loaded = _model().loaded
    return ("ok" if loaded else "model not loaded", 200 if loaded else 503)


@bp.route("/welcome")
def welcome() -> dict:
    """Service metadata, including execution providers once the model is loaded."""
    model = _model()
    config = _config()

    response = {
        "facerecognition-external-model": "InsightFace Edition",
        "version": PACKAGE_VERSION,
        "model": config.model_name,
        "model_status": "loaded" if model.loaded else "not loaded",
        "device": config.device,
    }

    if model.loaded:
        import onnxruntime as ort

        response["providers"] = ort.get_available_providers()

    return response
