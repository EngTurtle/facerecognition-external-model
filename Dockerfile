# ============================================================
# Declare ARG before any FROM
# ============================================================
ARG DEVICE=cpu

# ============================================================
# Builder variants
# ============================================================
FROM python:3.11-slim-bookworm AS builder-cpu
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements-cpu.txt /app/
RUN pip install --no-cache-dir -r requirements-cpu.txt
# --- Sanity check
RUN python3 -c "import sys, numpy; print('CPU builder OK:', sys.version, numpy.__version__)"

FROM builder-cpu AS builder-cuda
COPY requirements-cuda.txt /app/
RUN pip install --no-cache-dir -r requirements-cuda.txt
# --- Sanity check
RUN python3 -c "import onnx; print('CUDA builder OK:', onnx.__version__)"

FROM builder-cpu AS builder-openvino
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget ocl-icd-libopencl1 && \
    rm -rf /var/lib/apt/lists/*
COPY requirements-openvino.txt /app/
RUN pip install --no-cache-dir -r requirements-openvino.txt
# Install OpenVINO GPU deps directly here
RUN wget -nv https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17384.11/intel-igc-core_1.0.17384.11_amd64.deb && \
    wget -nv https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17384.11/intel-igc-opencl_1.0.17384.11_amd64.deb && \
    wget -nv https://github.com/intel/compute-runtime/releases/download/24.31.30508.7/intel-opencl-icd_24.31.30508.7_amd64.deb && \
    wget -nv https://github.com/intel/compute-runtime/releases/download/24.31.30508.7/libigdgmm12_22.4.1_amd64.deb && \
    dpkg -i *.deb && rm *.deb
# --- Sanity check
RUN test -f /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 && \
    python3 -c "import onnxruntime as ort; print('OpenVINO builder OK:', ort.get_device())"

# ============================================================
# Production variants
# ============================================================
FROM python:3.11-slim-bookworm AS prod-cpu
RUN python3 --version

FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04 AS prod-cuda
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.11 python3-pip libcudnn9-cuda-12 && \
    rm -rf /var/lib/apt/lists/*
RUN test -f /usr/local/cuda/lib64/libcudart.so.12

FROM python:3.11-slim-bookworm AS prod-openvino
RUN apt-get update && \
    apt-get install -y --no-install-recommends ocl-icd-libopencl1 && \
    rm -rf /var/lib/apt/lists/*
RUN test -f /usr/lib/x86_64-linux-gnu/libOpenCL.so.1

# ============================================================
# Selector stages
# ============================================================
FROM builder-${DEVICE} AS builder
FROM prod-${DEVICE} AS prod

# ============================================================
# Final assembly
# ============================================================
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY facerecognition_insightface.py gunicorn_config.py /app/
RUN mkdir -p /app/models /app/images
ARG DEVICE
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEVICE=$DEVICE \
    FLASK_APP=facerecognition_insightface.py \
    GUNICORN_WORKERS=1 \
    PORT=5000
EXPOSE 5000
HEALTHCHECK CMD curl -f http://localhost:5000/health || exit 1
ENTRYPOINT ["gunicorn", "-c", "gunicorn_config.py", "facerecognition_insightface:app"]
