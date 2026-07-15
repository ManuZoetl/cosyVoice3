# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG COSYVOICE_COMMIT=074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
ARG MODEL_ID=FunAudioLLM/Fun-CosyVoice3-0.5B-2512

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/huggingface \
    COSYVOICE_ROOT=/opt/CosyVoice \
    MODEL_DIR=/opt/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B \
    MODEL_NAME=fun-cosyvoice3-0.5b-2512 \
    VOICE_DATA_DIR=/workspace/cosyvoice-data \
    PORT=8000 \
    FP16=1 \
    ENABLE_SAMPLE_VOICE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
      ffmpeg \
      git \
      git-lfs \
      libsox-dev \
      python3-dev \
      sox \
      unzip \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /opt/CosyVoice \
    && cd /opt/CosyVoice \
    && git checkout "${COSYVOICE_COMMIT}" \
    && git submodule update --init --recursive

# The upstream requirements include training and TensorRT packages that are not
# needed for the default PyTorch streaming runtime. Torch is already supplied by
# the CUDA base image at the exact upstream version.
RUN python -m pip install --upgrade pip setuptools wheel \
    && grep -vE '^(deepspeed|tensorrt-cu12|tensorrt-cu12-bindings|tensorrt-cu12-libs|torch==|torchaudio==|gradio==|matplotlib==|tensorboard==|lightning==)' \
         /opt/CosyVoice/requirements.txt > /tmp/runtime-requirements.txt \
    && python -m pip install \
         --extra-index-url https://download.pytorch.org/whl/cu121 \
         --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
         -r /tmp/runtime-requirements.txt \
    && python -m pip install \
         huggingface_hub==0.30.2 \
         python-multipart==0.0.20 \
         websockets==15.0.1 \
    && rm -f /tmp/runtime-requirements.txt

RUN mkdir -p "${MODEL_DIR}" \
    && python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${MODEL_ID}', local_dir='${MODEL_DIR}')"

WORKDIR /app
COPY server.py /app/server.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh \
    && mkdir -p /workspace/cosyvoice-data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
  CMD curl --fail --silent http://127.0.0.1:${PORT}/health >/dev/null || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
