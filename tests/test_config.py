"""Unit tests for facerec.config (stdlib only)."""

import logging

import pytest

from facerec.config import (
    DEFAULT_API_KEY,
    DEFAULT_DET_SIZE,
    DEFAULT_EMBEDDING_SCALE,
    DEFAULT_MODEL_NAME,
    Config,
    load_api_key,
    parse_det_size,
)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("640,640", (640, 640)),
        ("1000,700", (992, 672)),
        ("800", (800, 800)),
        ("10,10", (32, 32)),
        ("garbage", (640, 640)),
    ],
)
def test_parse_det_size(value, expected):
    assert parse_det_size(value) == expected


def test_parse_det_size_default_matches_640_640():
    assert parse_det_size(DEFAULT_DET_SIZE) == (640, 640)


def test_config_from_env_custom_values():
    environ = {
        "MODEL_NAME": "antelopev2",
        "DEVICE": "CUDA",
        "MAX_DET_SIZE": "1000,700",
        "EMBEDDING_SCALE": "0.5",
        "API_KEY": "custom-key",
        "CUDA_DEVICE_ID": "1",
        "OPENVINO_DEVICE_TYPE": "GPU.1",
        "OPENVINO_PREC": "FP16",
    }

    config = Config.from_env(environ)

    assert config.model_name == "antelopev2"
    assert config.device == "cuda"  # lowercased
    assert config.det_size == (992, 672)
    assert config.embedding_scale == pytest.approx(0.5)
    assert config.api_key == "custom-key"
    assert config.cuda_device_id == "1"
    assert config.openvino_device_type == "GPU.1"
    assert config.openvino_precision == "FP16"


def test_config_from_env_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no api.key file present here

    config = Config.from_env({})

    assert config.model_name == DEFAULT_MODEL_NAME
    assert config.device == "cpu"
    assert config.det_size == (640, 640)
    assert config.embedding_scale == pytest.approx(DEFAULT_EMBEDDING_SCALE)
    assert config.api_key == DEFAULT_API_KEY
    assert config.cuda_device_id == "0"
    assert config.openvino_device_type == "GPU"
    assert config.openvino_precision == "FP32"


def test_load_api_key_env_wins(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "api.key").write_text("file-key\n")

    assert load_api_key({"API_KEY": "env-key"}) == "env-key"


def test_load_api_key_file_used_when_env_absent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "api.key").write_text("file-key\nwith-newline\n")

    assert load_api_key({}) == "file-keywith-newline"


def test_load_api_key_default_when_neither_exists(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)

    with caplog.at_level(logging.WARNING):
        key = load_api_key({})

    assert key == DEFAULT_API_KEY
    assert "insecure default" in caplog.text
