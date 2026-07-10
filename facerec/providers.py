"""Selection of ONNX Runtime execution providers based on Config.device."""

import logging

from facerec.config import Config

log = logging.getLogger(__name__)


def select_providers(config: Config) -> list:
    """Choose ONNX Runtime execution providers for the configured device.

    Imports onnxruntime lazily so this module stays importable in
    environments where onnxruntime is not installed.
    """
    import onnxruntime as ort

    available_providers = ort.get_available_providers()
    log.info("Available providers: %s", available_providers)

    if config.device == "openvino" and "OpenVINOExecutionProvider" in available_providers:
        try:
            openvino_device_ids = ort.capi._pybind_state.get_available_openvino_device_ids()
            log.info("Available OpenVINO devices: %s", openvino_device_ids)

            device_type = config.openvino_device_type

            if device_type not in openvino_device_ids:
                log.error(
                    "Requested device '%s' not found in available devices: %s",
                    device_type,
                    openvino_device_ids,
                )
                log.error("Falling back to CPU.")
                return ["CPUExecutionProvider"]

            if device_type.startswith("GPU"):
                precision = config.openvino_precision
                if precision not in ("FP32", "FP16"):
                    log.warning("Unsupported precision '%s' for Intel GPU. Defaulting to FP32.", precision)
                    precision = "FP32"

                openvino_options = {
                    "device_type": device_type,
                    "precision": precision,
                }
                log.info("Using OpenVINO with Intel GPU %s: %s", device_type, openvino_options)
            else:
                openvino_options = {
                    "device_type": device_type,
                }
                log.info("Using OpenVINO with device %s: %s", device_type, openvino_options)

            return [("OpenVINOExecutionProvider", openvino_options), "CPUExecutionProvider"]

        except Exception as e:
            log.error("Error detecting OpenVINO devices: %s", e)
            openvino_options = {
                "device_type": "CPU",
            }
            log.info("Using OpenVINO with CPU: %s", openvino_options)
            return [("OpenVINOExecutionProvider", openvino_options), "CPUExecutionProvider"]

    if config.device == "cuda" and "CUDAExecutionProvider" in available_providers:
        device_id = config.cuda_device_id
        cuda_options = {
            "arena_extend_strategy": "kSameAsRequested",
            "device_id": device_id,
        }
        log.info("Using CUDA with device %s: %s", device_id, cuda_options)
        return [("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"]

    provider_map = {
        "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "cpu": ["CPUExecutionProvider"],
    }

    preferred_providers = provider_map.get(config.device, ["CPUExecutionProvider"])

    providers = [p for p in preferred_providers if p in available_providers]

    if not providers:
        log.warning("Preferred providers %s not available, falling back to CPU", preferred_providers)
        providers = ["CPUExecutionProvider"]

    return providers
