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
docker-compose --profile cpu up -d facerecognition-cpu

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
| `EMBEDDING_SCALE` | `0.017` | Scaling the 512 dim face embedding to the range expected by clustering algorithm |
| `CUDA_DEVICE_ID` | `0` | CUDA device ID for multi-GPU systems |
| `OPENVINO_DEVICE_TYPE` | `GPU` | OpenVINO device: GPU, GPU.0, GPU.1, etc. (check logs for available options) |
| `OPENVINO_PREC` | `FP32` | OpenVINO precision for Intel GPU: FP32/FP16 |
| `MAX_DET_SIZE` | `640,640` | Detection input size as 'width,height' (snapped to multiples of 32) |
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

1. **Embeddings are incompatible** - InsightFace uses 512-dim embeddings vs dlib's 128-dim requiring changes to clustering sensitivity setting
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

# 5. Once confirmed working, stop old container
```

## Recommended Configuration Values

After implementing the InsightFace integration with 512-dimensional embeddings, the following settings are recommended for optimal face clustering:

### Admin Settings (Administration → Settings → Face Recognition)

| Setting | Recommended Value | Range | Notes |
|---------|------------------|-------|-------|
| **Sensitivity** | 0.4 | 0.2 - 0.6 | Middle value works well for most users |
| **Minimum confidence** | 0.7 | 0.5 - 0.95 | Keep at or above 0.7 |

### Understanding the Values

**Sensitivity:**

- **Lower (0.3-0.35):** Stricter matching - use if you have twins or very similar-looking people
- **Middle (0.4):** Balanced default - good for most photo libraries
- **Higher (0.45-0.5):** More permissive - catches marginal matches but may need manual cleanup

**Minimum confidence:**

- The **0.7** (default) is a good middle ground
- Going below 0.5 can introduce false positives
- Going above 0.9 will miss many valid faces

## Notes on recommended settings

These recommendations are based on [Immich's facial recognition documentation](https://immich.app/docs/features/facial-recognition), adapted for embedding scale difference between InsightFace and original dlib models.

- Immich uses normalized embeddings (cosine similarity) with thresholds of 0.3-0.7
- Face recognition uses raw Euclidean distance in the chinese whisper algorithm.
- The default `0.017` embedding scaling maps InsightFace's embedding distances (9-40) to facerecognition's UI setting range (0.2-0.6)
- The effective threshold behavior matches Immich's recommendations after scaling

For further guidance on improving clustering results, see [Immich's Better Facial Clusters guide](https://immich.app/docs/guides/better-facial-clusters).

## Testing Your Settings

After applying the code changes:

1. Start with **sensitivity = 0.4** and **min_faces = 2**
2. Re-run face detection and clustering on a sample of 100-1000 images
3. Review the results:
   - Too many separate clusters for the same person? → Increase sensitivity by 0.2 step each time
   - Unrelated people merged together? → Decrease sensitivity by 0.2 step each time
4. Once satisfied, apply to full library

## Device-Specific Notes

### CPU

- Works on any x86_64 system
- Consider using `buffalo_s` model for better speed

### CUDA

- Requires NVIDIA GPU with CUDA support
- Use `buffalo_l` or `antelopev2` for maximum accuracy

### OpenVINO

- Optimized for Intel hardware (iGPU and dGPU)
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
docker run ... -e MAX_DET_SIZE=1024,768 ...
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
