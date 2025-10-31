# DeepSeek-OCR 部署文档

## 📋 目录

- [系统要求](#系统要求)
- [本地部署](#本地部署)
- [Docker 部署](#docker-部署)
- [API 使用说明](#api-使用说明)
- [常见问题](#常见问题)

---

## 🖥️ 系统要求

### 硬件要求
- **GPU**: NVIDIA GPU (推荐 16GB+ 显存)
- **内存**: 32GB+ RAM (推荐 64GB)
- **存储**: 50GB+ 可用空间

### 软件要求
- **操作系统**: Windows 10/11, Linux (Ubuntu 20.04+)
- **CUDA**: 11.8+ (推荐 12.x)
- **Python**: 3.12.9
- **Docker**: 20.10+ (可选，用于容器化部署)
- **NVIDIA Container Toolkit**: 用于 Docker GPU 支持

---

## 🚀 本地部署

### 方法 1: 使用自动安装脚本 (Windows)

```powershell
# 运行安装脚本
.\setup_env.ps1
```

### 方法 2: 手动安装

#### 1. 创建 Conda 环境

```bash
# 创建环境
conda create -n deepseek-ocr python=3.12.9 -y

# 激活环境
conda activate deepseek-ocr
```

#### 2. 安装依赖

```bash
# 安装 PyTorch (CUDA 11.8)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# 安装基础依赖
pip install -r requirements.txt

# 安装 flash-attention
pip install flash-attn==2.7.3 --no-build-isolation

# 安装 API 服务依赖
pip install fastapi==0.109.0 uvicorn[standard]==0.27.0 python-multipart==0.0.6 aiofiles==23.2.1 requests
```

#### 3. 启动服务

```bash
# 启动 API 服务
python api_server.py
```

服务将在 `http://localhost:8000` 启动

#### 4. 测试服务

```bash
# 在新终端中运行测试
python test_api.py
```

---

## 🐳 Docker 部署

### 前置要求

1. **安装 Docker Desktop** (Windows)
   - 下载: https://www.docker.com/products/docker-desktop

2. **安装 NVIDIA Container Toolkit**
   ```bash
   # Linux
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

### 部署步骤

#### 1. 构建镜像

```bash
# 使用 docker-compose 构建
docker-compose build

# 或使用 docker 命令
docker build -t deepseek-ocr:latest .
```

#### 2. 启动服务

```bash
# 启动服务 (后台运行)
docker-compose up -d

# 查看日志
docker-compose logs -f
```

#### 3. 停止服务

```bash
# 停止服务
docker-compose down

# 停止并删除数据卷
docker-compose down -v
```

#### 4. 管理容器

```bash
# 查看容器状态
docker-compose ps

# 重启服务
docker-compose restart

# 进入容器
docker-compose exec deepseek-ocr bash
```

---

## 📡 API 使用说明

### API 文档

启动服务后，访问以下地址查看交互式 API 文档：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 端点说明

#### 1. 健康检查

```bash
GET /health
```

**响应示例:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "gpu_available": true,
  "timestamp": "2025-10-29T15:30:00"
}
```

#### 2. 图片 OCR

```bash
POST /ocr/image
```

**参数:**
- `file`: 图片文件 (multipart/form-data)
- `prompt`: OCR 提示词 (可选)
- `base_size`: 基础尺寸 (默认: 1024)
- `image_size`: 图片尺寸 (默认: 640)
- `crop_mode`: 裁剪模式 (默认: true)

**Python 示例:**
```python
import requests

url = "http://localhost:8000/ocr/image"
files = {'file': open('test.jpg', 'rb')}
data = {
    'prompt': '<image>\n<|grounding|>Convert the document to markdown.',
    'base_size': 1024,
    'image_size': 640,
    'crop_mode': True
}

response = requests.post(url, files=files, data=data)
result = response.json()
print(result['text'])
```

**cURL 示例:**
```bash
curl -X POST "http://localhost:8000/ocr/image" \
  -F "file=@test.jpg" \
  -F "prompt=<image>\n<|grounding|>Convert the document to markdown." \
  -F "base_size=1024" \
  -F "image_size=640" \
  -F "crop_mode=true"
```

#### 3. Base64 图片 OCR

```bash
POST /ocr/base64
```

**Python 示例:**
```python
import requests
import base64

with open('test.jpg', 'rb') as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

url = "http://localhost:8000/ocr/base64"
data = {
    'image_base64': image_base64,
    'prompt': '<image>\nFree OCR.'
}

response = requests.post(url, data=data)
result = response.json()
print(result['text'])
```

#### 4. 批量 OCR

```bash
POST /ocr/batch
```

**Python 示例:**
```python
import requests

url = "http://localhost:8000/ocr/batch"
files = [
    ('files', open('test1.jpg', 'rb')),
    ('files', open('test2.jpg', 'rb')),
    ('files', open('test3.jpg', 'rb'))
]
data = {'prompt': '<image>\n<|grounding|>Convert the document to markdown.'}

response = requests.post(url, files=files, data=data)
results = response.json()

for i, result in enumerate(results):
    if result['success']:
        print(f"图片 {i+1}: {result['text'][:100]}...")
```

### 提示词模板

根据不同场景选择合适的提示词：

```python
# 文档转 Markdown
prompt = "<image>\n<|grounding|>Convert the document to markdown."

# 通用 OCR
prompt = "<image>\n<|grounding|>OCR this image."

# 无布局识别
prompt = "<image>\nFree OCR."

# 图表解析
prompt = "<image>\nParse the figure."

# 详细描述
prompt = "<image>\nDescribe this image in detail."

# 定位文本
prompt = "<image>\nLocate <|ref|>目标文本<|/ref|> in the image."
```

---

## ❓ 常见问题

### 1. 模型下载慢或失败

**解决方案:**
```bash
# 设置 Hugging Face 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载模型
git lfs install
git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR
```

### 2. CUDA 内存不足

**解决方案:**
- 减小 `base_size` 和 `image_size` 参数
- 使用更小的批处理大小
- 关闭其他占用 GPU 的程序

### 3. Flash Attention 安装失败

**解决方案:**
```bash
# 确保安装了正确的 CUDA 工具链
pip install flash-attn==2.7.3 --no-build-isolation

# 如果仍然失败，可以跳过 flash-attention
# 修改 api_server.py 中的 _attn_implementation='eager'
```

### 4. Docker GPU 不可用

**解决方案:**
```bash
# 检查 NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# 确保 docker-compose.yml 中配置了 GPU
```

### 5. Windows 上 Docker 性能问题

**解决方案:**
- 使用 WSL2 后端
- 在 Docker Desktop 设置中分配足够的资源
- 考虑使用本地部署而非 Docker

---

## 📊 性能优化

### 1. 批处理优化

```python
# 使用批量接口处理多张图片
files = [('files', open(f'image_{i}.jpg', 'rb')) for i in range(10)]
response = requests.post(url, files=files)
```

### 2. 模型缓存

```bash
# 挂载模型缓存目录，避免重复下载
docker-compose.yml 中已配置:
volumes:
  - ./models:/root/.cache/huggingface
```

### 3. 并发处理

```python
# 使用多线程/多进程处理多个请求
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_image, img) for img in images]
    results = [f.result() for f in futures]
```

---

## 📞 支持

如有问题，请：
1. 查看日志: `docker-compose logs -f`
2. 检查 GPU 状态: `nvidia-smi`
3. 访问项目 GitHub: https://github.com/deepseek-ai/DeepSeek-OCR

---

## 📄 许可证

本项目遵循 MIT 许可证。详见 LICENSE 文件。

