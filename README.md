# Face Recognition External Model - InsightFace Edition

Improved version of [Face Recognition External Model](https://github.com/matiasdelellis/facerecognition-external-model) for Nextcloud, replacing dlib with InsightFace models following the Immich ML architecture.

## Why This Fork?

The original uses dlib models. This version uses InsightFace models which generally offer:

- More modern architecture (RetinaFace for detection, ArcFace for recognition)
- Multiple acceleration options (CPU, CUDA, OpenVINO)
- Larger embedding dimensions (512 vs 128)
- More recent training data

**Note**: Performance will vary based on your hardware and images. Test with your own data to evaluate.

## Features

### Multiple Acceleration Backends

- **CPU**: Works everywhere, good for small libraries
- **CUDA**: NVIDIA GPU acceleration
- **OpenVINO**: Intel GPU/CPU optimization

### Available Models

- `buffalo_s` - Small/fastest
- `buffalo_l` - Large (default)
- `antelopev2` - Alternative high-accuracy model

### API Compatibility

Maintains full compatibility with Nextcloud Face Recognition app:

- Same endpoints: `/detect`, `/compute`, `/open`, `/health`, `/welcome`
- Same request/response format
- Drop-in replacement

## Installation

### Prerequisites

```bash
# Install Docker and Docker Compose
# For GPU support:
#   - CUDA: Install nvidia-docker2
#   - OpenVINO: Ensure /dev/dri device is available
```

### Using Pre-built Images

Pre-built images are available from GitHub Container Registry:

```bash
# CPU
docker run -d --name facerecognition \
  -p 8080:5000 \
  -e API_KEY="some-super-secret-api-key" \
  -e MODEL_NAME="buffalo_l" \
  -e DEVICE="cpu" \
  -v model-cache:/app/models \
  ghcr.io/engturtle/facerecognition-external-model-insightface:cpu

# CUDA
docker run -d --name facerecognition \
  --gpus all \
  -p 8080:5000 \
  -e API_KEY="some-super-secret-api-key" \
  -e MODEL_NAME="buffalo_l" \
  -e DEVICE="cuda" \
  -v model-cache:/app/models \
  ghcr.io/engturtle/facerecognition-external-model-insightface:cuda

# OpenVINO
docker run -d --name facerecognition \
  --device /dev/dri:/dev/dri \
  -p 8080:5000 \
  -e API_KEY="some-super-secret-api-key" \
  -e MODEL_NAME="buffalo_l" \
  -e DEVICE="openvino" \
  -v model-cache:/app/models \
  ghcr.io/engturtle/facerecognition-external-model-insightface:openvino
```

### Build Locally (Optional)

```bash
# Build
docker build --build-arg DEVICE=cpu -t facerecognition-insightface:cpu -f Dockerfile .
docker build --build-arg DEVICE=cuda -t facerecognition-insightface:cuda -f Dockerfile .
docker build --build-arg DEVICE=openvino -t facerecognition-insightface:openvino -f Dockerfile .

# Run (using local builds)
docker run -d --name facerecognition \
  -p 8080:5000 \
  -e API_KEY="some-super-secret-api-key" \
  -e MODEL_NAME="buffalo_l" \
  -e DEVICE="cpu" \
  -v model-cache:/app/models \
  facerecognition-insightface:cpu
```

### Using Docker Compose

```bash
# CPU (default)
docker-compose up -d facerecognition-cpu

# CUDA
docker-compose --profile cuda up -d facerecognition-cuda

# OpenVINO
docker-compose --profile openvino up -d facerecognition-openvino
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | `some-super-secret-api-key` | Shared API key for authentication |
| `MODEL_NAME` | `buffalo_l` | InsightFace model: buffalo_s/l or antelopev2 |
| `DEVICE` | `cpu` | Acceleration backend: cpu/cuda/openvino |
| `CUDA_DEVICE_ID` | `0` | CUDA device ID for multi-GPU systems |
| `OPENVINO_DEVICE_TYPE` | `GPU` | OpenVINO device: GPU, GPU.0, GPU.1, etc. (check logs for available options) |
| `OPENVINO_PREC` | `FP32` | OpenVINO precision for Intel GPU: FP32/FP16 |
| `MAX_DET_SIZE` | `2048` | Detection input size, max of width and height |
| `GUNICORN_WORKERS` | `1` | Number of worker processes |
| `REQ_TIMEOUT` | `300` | Request timeout in seconds |
| `PRELOAD_MODELS` | `false` | Preload models at startup |
| `PORT` | `5000` | Server port |

### Generate Secure API Key

```bash
openssl rand -base64 32 > api.key
```

## Nextcloud Configuration

```bash
# Set external model URL
php occ config:system:set facerecognition.external_model_url \
  --value "http://your-server:8080"

# Set API key
php occ config:system:set facerecognition.external_model_api_key \
  --value "some-super-secret-api-key"

# Enable external model (model 5)
php occ face:setup -m 5

# Verify
php occ face:stats
```

## Testing

```bash
# Test service
curl http://localhost:8080/welcome

# Should return:
# {
#   "facerecognition-external-model": "InsightFace Edition",
#   "version": "2.0.0",
#   "model": "buffalo_l",
#   "device": "cpu",
#   "providers": ["CPUExecutionProvider"]
# }

# Test detection with an image
curl -X POST http://localhost:8080/detect \
  -H "x-api-key: some-super-secret-api-key" \
  -F "file=@test.jpg"
```

## Migration from Original

### Important Notes

1. **Embeddings are incompatible** - InsightFace uses 512-dim embeddings vs dlib's 128-dim
2. **Must regenerate face data** - All faces need to be re-analyzed
3. **API is compatible** - No changes needed to Nextcloud configuration except URL/key

### Migration Steps

```bash
# 1. Start new container (on different port initially)
docker run -d --name facerecognition-new \
  -p 5000:5000 \
  -e API_KEY="new-api-key" \
  facerecognition-insightface:cpu

# 2. Test new container
curl http://localhost:5000/welcome

# 3. Update Nextcloud configuration
php occ config:system:set facerecognition.external_model_url \
  --value "http://your-server:5000"
php occ config:system:set facerecognition.external_model_api_key \
  --value "new-api-key"

# 4. Reset and regenerate face data
php occ face:reset --all
php occ face:background_job --all --analyze-mode

# 5. Once confirmed working, stop old container and switch to port 8080
```

## Device-Specific Notes

### CPU

- Works on any x86_64 system
- Consider using `buffalo_s` model for better speed

### CUDA

- Requires NVIDIA GPU with CUDA support
- Requires nvidia-docker2
- Use `buffalo_l` or `antelopev2` for maximum accuracy

### OpenVINO

- Optimized for Intel hardware (iGPU and dGPU)
- Good middle ground between CPU and CUDA
- Requires `/dev/dri` device access for GPU support
- Works on CPU-only mode as well

## Troubleshooting

### Models Not Downloading

Models download on first use. Check logs:

```bash
docker logs facerecognition
```

For manual download:

```bash
docker exec facerecognition python3 -c \
  "from insightface.app import FaceAnalysis; \
   FaceAnalysis(name='buffalo_l', root='/app/models')"
```

### Wrong Execution Provider

Check which provider is actually being used:

```bash
curl http://localhost:8080/welcome | jq '.providers'
```

Expected outputs:

- CPU: `["CPUExecutionProvider"]`
- CUDA: `["CUDAExecutionProvider", "CPUExecutionProvider"]`
- OpenVINO: `["OpenVINOExecutionProvider", "CPUExecutionProvider"]`

### Out of Memory

```bash
# Reduce workers
docker run ... -e GUNICORN_WORKERS=1 ...

# Use smaller model
docker run ... -e MODEL_NAME=buffalo_s ...

# Reduce detection size
docker run ... -e DET_SIZE=1024 ...
```

### Slow Performance

1. Check if using intended device:

   ```bash
   docker logs facerecognition | grep "Using device"
   ```

2. For CPU: Use smaller model or reduce detection size
3. For GPU: Ensure GPU is actually being used (check nvidia-smi or intel_gpu_top)
4. For OpenVINO: Verify GPU device is accessible

### Local Development

```bash
# Install dependencies
pip install -r requirements-cpu.txt

# Run locally
export API_KEY="test-key"
export MODEL_NAME="buffalo_l"
export DEVICE="cpu"
python facerecognition_insightface.py
```

## Architecture

Following Immich's ML architecture:

- Multi-stage Docker builds for different acceleration backends
- ONNX Runtime with pluggable execution providers
- Device-aware provider selection
- Graceful fallback to CPU if preferred provider unavailable

## Acknowledgments

- Original project: [facerecognition-external-model](https://github.com/matiasdelellis/facerecognition-external-model) by Matias De lellis
- InsightFace: [InsightFace](https://github.com/deepinsight/insightface) by DeepInsight
- Immich ML: [Immich](https://github.com/immich-app/immich) for architecture inspiration

## License

Same license as original project. InsightFace models used according to their respective licenses.
