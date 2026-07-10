"""Unit tests for facerec.faces (numpy + stdlib only)."""

import types

import numpy as np
import pytest

from facerec.faces import best_face_match, iou, serialize_face, serialize_landmarks


def _face(bbox, det_score=0.9, kps=None, embedding=None):
    return types.SimpleNamespace(
        bbox=np.array(bbox, dtype=np.float64),
        kps=kps,
        det_score=np.float32(det_score),
        embedding=np.array(embedding if embedding is not None else [1.0, 2.0, 3.0, 4.0]),
    )


def test_iou_identical_boxes():
    box = [0, 0, 10, 10]
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_disjoint_boxes():
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_iou_partial_overlap():
    # box1: [0,0,10,10] area 100; box2: [5,5,15,15] area 100
    # intersection: [5,5,10,10] area 25; union: 100+100-25=175
    assert iou([0, 0, 10, 10], [5, 5, 15, 15]) == pytest.approx(25 / 175)


def test_iou_zero_area_boxes():
    assert iou([0, 0, 0, 0], [0, 0, 10, 10]) == 0.0


def test_serialize_landmarks_none():
    assert serialize_landmarks(None) == []


def test_serialize_landmarks_points():
    kps = np.array([[1.4, 2.6], [3.0, 4.0]])
    assert serialize_landmarks(kps) == [{"x": 1, "y": 2}, {"x": 3, "y": 4}]


def test_serialize_face():
    face = _face(
        bbox=[10.4, 20.6, 100.2, 200.9],
        det_score=0.87,
        kps=np.array([[15.0, 25.0], [30.0, 40.0]]),
        embedding=[1.0, 2.0, 3.0, 4.0],
    )

    result = serialize_face(face, embedding_scale=0.017)

    assert set(result.keys()) == {
        "detection_confidence",
        "left",
        "top",
        "right",
        "bottom",
        "landmarks",
        "descriptor",
    }
    assert result["detection_confidence"] == pytest.approx(0.87, abs=1e-6)
    assert isinstance(result["detection_confidence"], float)
    assert result["left"] == 10
    assert result["top"] == 20
    assert result["right"] == 100
    assert result["bottom"] == 200
    assert all(isinstance(v, int) for v in (result["left"], result["top"], result["right"], result["bottom"]))
    assert result["landmarks"] == [{"x": 15, "y": 25}, {"x": 30, "y": 40}]
    assert result["descriptor"] == pytest.approx([0.017, 0.034, 0.051, 0.068])


def test_serialize_face_no_landmarks():
    face = _face(bbox=[0, 0, 10, 10], kps=None)
    result = serialize_face(face, embedding_scale=1.0)
    assert result["landmarks"] == []


def test_best_face_match_picks_highest_iou():
    target_box = [0, 0, 10, 10]
    low_overlap = _face(bbox=[8, 8, 20, 20])   # small overlap
    high_overlap = _face(bbox=[0, 0, 10, 10])  # identical, IoU 1.0
    faces = [low_overlap, high_overlap]

    assert best_face_match(faces, target_box) is high_overlap


def test_best_face_match_empty_list():
    assert best_face_match([], [0, 0, 10, 10]) is None


def test_best_face_match_no_overlap():
    faces = [_face(bbox=[100, 100, 110, 110])]
    assert best_face_match(faces, [0, 0, 10, 10]) is None
