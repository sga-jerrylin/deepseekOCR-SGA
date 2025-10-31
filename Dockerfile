# DeepSeek-OCR Docker Image
# 基于 NVIDIA CUDA 12.6 镜像，支持 RTX 50 系列 GPU (sm_120)

FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/opt/conda/bin:$PATH \
    MODEL_NAME=deepseek-ai/DeepSeek-OCR \
    OUTPUT_DIR=/app/outputs

# 设置工作目录
WORKDIR /app

# 更换为清华大学镜像源（解决网络问题）
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list && \
    sed -i 's@//.*security.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 Miniforge（使用 conda-forge，无需 TOS）
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh && \
    /opt/conda/bin/conda clean -a -y

# 创建 conda 环境
RUN conda create -n deepseek-ocr python=3.12.9 -y && \
    conda clean -a -y

# 激活环境
SHELL ["conda", "run", "-n", "deepseek-ocr", "/bin/bash", "-c"]

# 复制项目文件
COPY requirements.txt /app/
COPY api_server.py /app/
COPY DeepSeek-OCR-master /app/DeepSeek-OCR-master

# 安装 PyTorch nightly with CUDA 12.8 (支持 RTX 50 系列 sm_120)
RUN pip install --no-cache-dir --pre \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# 安装其他依赖
RUN pip install --no-cache-dir -r requirements.txt

# 注意：不安装 flash-attention，使用 eager 模式（避免长时间编译）
# Flash Attention 编译需要 15-60 分钟，但不是必需的
# 我们在 api_server.py 中使用 attn_implementation='eager'

# 安装 FastAPI 和相关依赖
RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn[standard]==0.27.0 \
    python-multipart==0.0.6 \
    aiofiles==23.2.1

# 创建输出目录
RUN mkdir -p /app/outputs

# 暴露端口
EXPOSE 8200

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8200/health || exit 1

# 启动命令
CMD ["conda", "run", "--no-capture-output", "-n", "deepseek-ocr", \
     "python", "api_server.py"]

