FROM python:slim AS builder

COPY Makefile /app/

# Install optimized dependencies with OpenBLAS
RUN apt update -yq \
    && apt install -yq \
        bzip2 cmake g++ make wget \
        libopenblas-dev \
        liblapack-dev \
        libblas-dev \
        gfortran \
        pkg-config \
        python3-dev \
    && pip install numpy \
    && OPENBLAS_NUM_THREADS=4 DLIB_YES_I_UNDERSTAND_THIS_IS_UNOFFICIAL=1 pip wheel -w /app/ dlib \
    && make -C /app/ download-models

FROM python:slim

COPY --from=builder /app/dlib*.whl /tmp/
COPY --from=builder /app/vendor/ /app/vendor/

# Install OpenBLAS runtime  
RUN apt update -yq \
    && apt install -yq \
        libopenblas0 \
        liblapack3 \
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
    OPENBLAS_MAIN_FREE=0

ENTRYPOINT ["gunicorn", "-c", "gunicorn_config.py", "facerecognition-external-model:app"]