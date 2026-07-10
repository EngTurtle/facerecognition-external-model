"""Pure face-geometry and serialization helpers (numpy + stdlib only)."""

from typing import Any, Iterable, Optional, Sequence


def iou(box1: Sequence[int], box2: Sequence[int]) -> float:
    """Intersection over union of two [left, top, right, bottom] boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0


def serialize_landmarks(kps) -> list[dict[str, int]]:
    """Convert an Nx2 keypoint array (or None) to a list of {x, y} dicts."""
    if kps is None:
        return []
    return [{"x": int(point[0]), "y": int(point[1])} for point in kps]


def serialize_face(face, embedding_scale: float) -> dict[str, Any]:
    """Convert an InsightFace detection into the Nextcloud-compatible dict."""
    bbox = face.bbox.astype(int)
    embedding = face.embedding * embedding_scale

    return {
        "detection_confidence": float(face.det_score),
        "left": int(bbox[0]),
        "top": int(bbox[1]),
        "right": int(bbox[2]),
        "bottom": int(bbox[3]),
        "landmarks": serialize_landmarks(face.kps),
        "descriptor": embedding.tolist(),
    }


def best_face_match(faces: Iterable, target_box: Sequence[int]) -> Optional[Any]:
    """Return the face whose bbox has the highest positive IoU with target_box."""
    best_match = None
    best_overlap = 0.0

    for face in faces:
        bbox = face.bbox.astype(int)
        overlap = iou(target_box, bbox.tolist())
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = face

    return best_match
