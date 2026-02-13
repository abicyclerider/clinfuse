# MedGemma entity resolution pipeline.
# Build: docker build -t medgemma .
# Run:   docker run --gpus all -e HF_TOKEN=hf_... medgemma python <script>.py [args]

FROM nvidia/cuda:12.4.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA 12.4 support (needs custom index URL, so separate from requirements)
RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install all Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY prepare_base_model.py \
     train_classifier_on_gpu.py \
     export_text_only_model.py \
     inference_classifier.py \
     ./

CMD ["sleep", "infinity"]
