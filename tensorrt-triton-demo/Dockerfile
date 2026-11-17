FROM nvcr.io/nvidia/tritonserver:25.10-trtllm-python-py3

ENV PYTHONPATH=/usr/local/lib/python3.12/dist-packages:$PYTHONPATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/persistent-storage/models
ENV TORCH_CUDA_ARCH_LIST=8.6

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --break-system-packages \
    huggingface_hub \
    transformers

# Create directories
RUN mkdir -p \
    /app/model_repository/llama3_2/1 \
    /persistent-storage/models \
    /persistent-storage/engines

# Copy files
COPY model.py /app/model_repository/llama3_2/1/
COPY config.pbtxt /app/model_repository/llama3_2/

EXPOSE 8000 8001 8002

CMD ["tritonserver", "--model-repository=/app/model_repository", "--http-port=8000", "--grpc-port=8001", "--metrics-port=8002"]
