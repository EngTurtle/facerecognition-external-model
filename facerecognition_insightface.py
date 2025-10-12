from typing import List, Dict, Any
from flask import Flask, request, abort
from functools import wraps
import os
import json
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# Info
PACKAGE_VERSION = "0.1.0"

# Model globals
face_app: FaceAnalysis = None
face_model = None
MODEL_NAME = os.environ.get("MODEL_NAME", "buffalo_l")

# Directory for temporary images
TEMP_DIR = "images"

app = Flask(__name__)


MAX_DET_SIZE = int(os.environ.get("MAX_DET_SIZE", 2048))
# Ensure multiple of 32
MAX_DET_SIZE = (MAX_DET_SIZE // 32) * 32

def compute_det_size(img_shape):
    """Scale image to fit within MAX_DET_SIZE, rounded to multiple of 32."""
    h, w = img_shape[:2]
    scale = min(MAX_DET_SIZE / max(h, w), 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    new_w = max((new_w // 32) * 32, 32)
    new_h = max((new_h // 32) * 32, 32)
    return (new_w, new_h)


def require_appkey(view_function):
    """Security decorator for API key authentication"""
    @wraps(view_function)
    def decorated_function(*args, **kwargs):
        if 'API_KEY' in os.environ:
            key = os.environ.get('API_KEY')
        else:
            try:
                with open('api.key', 'r') as apikey:
                    key = apikey.read().replace('\n', '')
            except FileNotFoundError:
                key = 'some-super-secret-api-key'
        
        if request.headers.get('x-api-key') and request.headers.get('x-api-key') == key:
            return view_function(*args, **kwargs)
        else:
            abort(401)
    
    return decorated_function


def load_insightface_models():
    """Load InsightFace models for face detection and recognition"""
    global face_app, face_model
    
    if face_app is not None:
        return
    
    print(f"Loading InsightFace model: {MODEL_NAME}")
    
    # Determine execution providers based on device
    device = os.environ.get('DEVICE', 'cpu').lower()
    providers = get_providers(device)
    
    print(f"Using device: {device}")
    print(f"Execution providers: {providers}")
    
    # Initialize FaceAnalysis app for detection
    face_app = FaceAnalysis(
        name=MODEL_NAME,
        root='./models',
        allowed_modules=['detection', 'recognition'],
        providers=providers
    )
    
    # Prepare with context
    # For OpenVINO/CUDA, we use ctx_id=0, for CPU we use -1
    ctx_id = 0 if device in ['cuda', 'openvino'] else -1
    det_size = (MAX_DET_SIZE, MAX_DET_SIZE)
    face_app.prepare(ctx_id=ctx_id, det_size=det_size)
    
    print(f"InsightFace model loaded successfully")


def get_providers(device: str) -> list:
    """Get ONNX Runtime execution providers based on device"""
    import onnxruntime as ort
    import os
    
    available_providers = ort.get_available_providers()
    print(f"Available providers: {available_providers}")
    
    # Configure OpenVINO for Intel GPU
    if device == 'openvino' and 'OpenVINOExecutionProvider' in available_providers:
        try:
            # Use ONNX Runtime's built-in OpenVINO device detection
            openvino_device_ids = ort.capi._pybind_state.get_available_openvino_device_ids()
            print(f"Available OpenVINO devices: {openvino_device_ids}")
            
            # Get device configuration
            device_type = os.environ.get('OPENVINO_DEVICE_TYPE', 'GPU')
            
            # Validate device_type is actually available
            if device_type not in openvino_device_ids:
                print(f"Error: Requested device '{device_type}' not found in available devices: {openvino_device_ids}")
                print(f"Falling back to CPU.")
                return ['CPUExecutionProvider']
            
            # Configure based on device type
            if device_type.startswith('GPU'):
                # GPU configuration with precision options
                precision = os.environ.get('OPENVINO_PREC', 'FP32')
                if precision not in ['FP32', 'FP16']:
                    print(f"Warning: Unsupported precision '{precision}' for Intel GPU. Defaulting to FP32.")
                    precision = 'FP32'
                
                openvino_options = {
                    'device_type': device_type,
                    'precision': precision,
                }
                print(f"Using OpenVINO with Intel GPU {device_type}: {openvino_options}")
            else:
                # CPU or other device - no precision options (untested)
                openvino_options = {
                    'device_type': device_type,
                }
                print(f"Using OpenVINO with device {device_type}: {openvino_options}")
            
            return [('OpenVINOExecutionProvider', openvino_options), 'CPUExecutionProvider']
                
        except Exception as e:
            print(f"Error detecting OpenVINO devices: {e}")
            # Fall back to OpenVINO CPU if device detection fails
            openvino_options = {
                'device_type': 'CPU',
            }
            print(f"Using OpenVINO with CPU: {openvino_options}")
            return [('OpenVINOExecutionProvider', openvino_options), 'CPUExecutionProvider']
    
    # Configure CUDA provider with options (similar to Immich ML)
    if device == 'cuda' and 'CUDAExecutionProvider' in available_providers:
        device_id = os.environ.get('CUDA_DEVICE_ID', '0')
        cuda_options = {
            'arena_extend_strategy': 'kSameAsRequested',
            'device_id': device_id
        }
        print(f"Using CUDA with device {device_id}: {cuda_options}")
        return [('CUDAExecutionProvider', cuda_options), 'CPUExecutionProvider']
    
    # Define provider preference based on device
    provider_map = {
        'cuda': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
        'cpu': ['CPUExecutionProvider']
    }
    
    preferred_providers = provider_map.get(device, ['CPUExecutionProvider'])
    
    # Only use providers that are actually available
    providers = [p for p in preferred_providers if p in available_providers]
    
    if not providers:
        print(f"Warning: Preferred providers {preferred_providers} not available, falling back to CPU")
        providers = ['CPUExecutionProvider']
    
    return providers


def image_to_numpy(image_path: str) -> np.ndarray:
    """Load image from path and convert to numpy array"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def serialize_face(face) -> Dict[str, Any]:
    """Convert InsightFace detection to Nextcloud-compatible format"""
    bbox = face.bbox.astype(int)
    
    # InsightFace uses landmarks in different order, reformat for compatibility
    landmarks = []
    if face.kps is not None:
        for point in face.kps:
            landmarks.append({"x": int(point[0]), "y": int(point[1])})
    
    # Serialize embedding as list of floats
    embedding = face.normed_embedding.tolist() if hasattr(face, 'normed_embedding') else face.embedding.tolist()
    
    return {
        "detection_confidence": float(face.det_score),
        "left": int(bbox[0]),
        "top": int(bbox[1]),
        "right": int(bbox[2]),
        "bottom": int(bbox[3]),
        "landmarks": landmarks,
        "descriptor": embedding
    }


@app.route("/detect", methods=["POST"])
@require_appkey
def detect_faces() -> dict:
    """Detect faces in uploaded image"""
    uploaded_file = request.files.get("file")
    if not uploaded_file:
        abort(400, "No file provided")
    
    filename = os.path.basename(uploaded_file.filename)
    image_path = os.path.join(TEMP_DIR, filename)
    
    try:
        # Save uploaded file
        uploaded_file.save(image_path)
        
        # Load image
        img = image_to_numpy(image_path)
        
        # Check image size
        if max(img.shape[0], img.shape[1]) > MAX_DET_SIZE:
            abort(412, "Image too large")
        
        # Ensure models are loaded
        if face_app is None:
            load_insightface_models()
        
        # Get minimum score threshold from request options, or use default
        # Nextcloud sends this in the request,
        min_score = 0.5  # Default if not specified
        try:
            # The options are sent as form data in Nextcloud's request
            # Format varies, check both possible locations
            if 'minScore' in request.form:
                min_score = float(request.form.get('minScore', 0.5))
            elif 'options' in request.form:
                options = json.loads(request.form.get('options', '{}'))
                min_score = float(options.get('minScore', 0.5))
        except (ValueError, json.JSONDecodeError):
            pass  # Use default
        
        # Detect faces
        faces = face_app.get(img)
        
        # Filter by confidence score (from request or default)
        faces = [face for face in faces if face.det_score >= min_score]
        
        # Serialize faces
        serialized_faces = [serialize_face(face) for face in faces]
        
        return {
            "filename": filename,
            "faces-count": len(serialized_faces),
            "faces": serialized_faces
        }
    
    finally:
        # Clean up temporary file
        if os.path.exists(image_path):
            os.remove(image_path)


@app.route("/compute", methods=["POST"])
@require_appkey
def compute():
    """Compute face embedding for a specific face region"""
    uploaded_file = request.files.get("file")
    face_json_str = request.form.get("face")
    
    if not uploaded_file or not face_json_str:
        abort(400, "Missing file or face data")
    
    try:
        face_data = json.loads(face_json_str)
    except json.JSONDecodeError:
        abort(400, "Invalid face JSON")
    
    filename = os.path.basename(uploaded_file.filename)
    image_path = os.path.join(TEMP_DIR, filename)
    
    try:
        # Save uploaded file
        uploaded_file.save(image_path)
        
        # Load image
        img = image_to_numpy(image_path)
        
        # Check image size
        if max(img.shape[0], img.shape[1]) > MAX_DET_SIZE:
            abort(412, "Image too large")
        
        # Ensure models are loaded
        if face_app is None:
            load_insightface_models()
        
        # Detect all faces and find the one matching the bounding box
        faces = face_app.get(img)
        
        # Find face closest to provided bounding box
        target_bbox = [
            face_data.get("left", 0),
            face_data.get("top", 0),
            face_data.get("right", 0),
            face_data.get("bottom", 0)
        ]
        
        best_match = None
        best_overlap = 0
        
        for face in faces:
            bbox = face.bbox.astype(int)
            overlap = calculate_iou(target_bbox, bbox.tolist())
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = face
        
        if best_match is None:
            # If no match found, return original face_data with empty descriptor
            face_data["descriptor"] = []
            face_data["landmarks"] = []
            return {"filename": filename, "face": face_data}
        
        # Update face data with computed values
        if best_match.kps is not None:
            landmarks = []
            for point in best_match.kps:
                landmarks.append({"x": int(point[0]), "y": int(point[1])})
            face_data["landmarks"] = landmarks
        
        embedding = (best_match.normed_embedding if hasattr(best_match, 'normed_embedding') 
                    else best_match.embedding)
        face_data["descriptor"] = embedding.tolist()
        
        return {"filename": filename, "face": face_data}
    
    finally:
        # Clean up temporary file
        if os.path.exists(image_path):
            os.remove(image_path)


def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """Calculate Intersection over Union for two bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0


@app.route("/open")
@require_appkey
def open_model():
    """Pre-load models and return configuration"""
    load_insightface_models()
    return {
        "preferred_mimetype": "image/jpeg",
        "maximum_area": MAX_DET_SIZE^2,
        "model": MODEL_NAME
    }


@app.route("/health")
def health():
    """Health check endpoint"""
    return 'ok'


@app.route("/welcome")
def welcome():
    """Welcome endpoint with version info"""
    model_status = "loaded" if face_app is not None else "not loaded"
    device = os.environ.get('DEVICE', 'cpu')
    
    response = {
        "facerecognition-external-model": "InsightFace Edition",
        "version": PACKAGE_VERSION,
        "model": MODEL_NAME,
        "model_status": model_status,
        "device": device
    }
    
    # Add provider info if model is loaded
    if face_app is not None:
        try:
            import onnxruntime as ort
            response["providers"] = ort.get_available_providers()
        except:
            pass
    
    return response


if __name__ == "__main__":
    # Ensure temp directory exists
    os.makedirs(TEMP_DIR, exist_ok=True)
    app.run(host='0.0.0.0', port=5000)
