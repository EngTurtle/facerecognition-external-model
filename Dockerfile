FROM python:slim AS builder

COPY Makefile /app/

# Enable non-free repositories for Intel MKL
RUN sed -i 's/Components: main$/Components: main contrib non-free non-free-firmware/' /etc/apt/sources.list.d/debian.sources

# Pre-configure MKL installation to avoid interactive prompts
RUN echo "intel-mkl intel-mkl/use_intel_mkl boolean true" | debconf-set-selections \
    && echo "intel-mkl intel-mkl/alternatives_list multiselect 1, 2" | debconf-set-selections

# Install optimized dependencies with Intel MKL
RUN apt update -yq \
    && DEBIAN_FRONTEND=noninteractive apt install -yq \
        bzip2 cmake g++ make wget \
        intel-mkl \
        gfortran \
        pkg-config \
        python3-dev \
    && pip install numpy \
    && DLIB_YES_I_UNDERSTAND_THIS_IS_UNOFFICIAL=1 pip wheel -w /app/ dlib \
    && make -C /app/ download-models

FROM python:slim

# Enable non-free repositories for runtime
RUN sed -i 's/Components: main$/Components: main contrib non-free non-free-firmware/' /etc/apt/sources.list.d/debian.sources

# Pre-configure MKL for runtime
RUN echo "intel-mkl intel-mkl/use_intel_mkl boolean true" | debconf-set-selections \
    && echo "intel-mkl intel-mkl/alternatives_list multiselect 1, 2" | debconf-set-selections

COPY --from=builder /app/dlib*.whl /tmp/
COPY --from=builder /app/vendor/ /app/vendor/

# Install Intel MKL runtime
RUN apt update -yq \
    && DEBIAN_FRONTEND=noninteractive apt install -yq \
        intel-mkl \
    && pip install flask numpy gunicorn \
    && pip install --no-index -f /tmp/ dlib \
    && rm /tmp/dlib*.whl \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

COPY facerecognition-external-model.py /app/
COPY gunicorn_config.py /app/

WORKDIR /app/

EXPOSE 5000

ARG GUNICORN_WORKERS="1" \
    PORT="5000"
ENV GUNICORN_WORKERS="${GUNICORN_WORKERS}"\
    PORT="${PORT}"\
    API_KEY=some-super-secret-api-key\
    FLASK_APP=facerecognition-external-model.py\
    OMP_NUM_THREADS=4\
    MKL_NUM_THREADS=4\
    MKL_DOMAIN_NUM_THREADS="MKL_BLAS=4,MKL_FFT=1"

ENTRYPOINT ["gunicorn", "-c", "gunicorn_config.py", "facerecognition-external-model:app"]