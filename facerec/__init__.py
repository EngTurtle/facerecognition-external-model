"""facerec: InsightFace-backed face detection/recognition service for Nextcloud.

Submodules are intentionally not imported here so that lightweight modules
(e.g. `facerec.config`) remain importable in environments without
insightface/onnxruntime/cv2 installed.
"""

PACKAGE_VERSION = "0.1.0"
