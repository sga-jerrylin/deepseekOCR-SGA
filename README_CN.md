# DeepSeek-OCR 本地部署指南

## 🎯 快速开始

### 方式一：本地部署（推荐用于开发测试）

#### 1. 安装环境

双击运行 `install.bat` 或在命令行中执行：

```bash
install.bat
```

这将自动完成：
- 创建 Python 3.12.9 的 conda 环境
- 安装 PyTorch 2.6.0 (CUDA 11.8)
- 安装所有依赖包
- 创建必要的目录

#### 2. 启动服务

双击运行 `start_server.bat` 或在命令行中执行：

```bash
start_server.bat
```

服务将在 `http://localhost:8000` 启动

#### 3. 访问 API 文档

在浏览器中打开：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

#### 4. 测试 API

```bash
conda activate deepseek-ocr
python test_api.py
```

---

### 方式二：Docker 部署（推荐用于生产环境）

#### 前置要求

1. 安装 Docker Desktop
2. 安装 NVIDIA Container Toolkit（用于 GPU 支持）

#### 部署步骤

```bash
# 1. 构建镜像
docker-compose build

# 2. 启动服务
docker-compose up -d

# 3. 查看日志
docker-compose logs -f

# 4. 停止服务
docker-compose down
```

---

## 📡 API 使用示例

### Python 调用示例

```python
import requests

# 1. 健康检查
response = requests.get("http://localhost:8000/health")
print(response.json())

# 2. 图片 OCR
url = "http://localhost:8000/ocr/image"
files = {'file': open('test.jpg', 'rb')}
data = {
    'prompt': '<image>\n<|grounding|>Convert the document to markdown.',
    'base_size': 1024,
    'image_size': 640
}
response = requests.post(url, files=files, data=data)
result = response.json()
print(result['text'])
```

### cURL 调用示例

```bash
# 健康检查
curl http://localhost:8000/health

# 图片 OCR
curl -X POST "http://localhost:8000/ocr/image" \
  -F "file=@test.jpg" \
  -F "prompt=<image>\n<|grounding|>Convert the document to markdown."
```

---

## 🔧 配置说明

### 环境变量

在 `api_server.py` 中可以配置：

```python
MODEL_NAME = "deepseek-ai/DeepSeek-OCR"  # 模型名称
OUTPUT_DIR = "./outputs"                  # 输出目录
```

### 提示词模板

根据不同场景选择：

```python
# 文档转 Markdown
"<image>\n<|grounding|>Convert the document to markdown."

# 通用 OCR
"<image>\n<|grounding|>OCR this image."

# 无布局识别
"<image>\nFree OCR."

# 图表解析
"<image>\nParse the figure."
```

---

## 📊 性能参数

### 图片尺寸配置

- `base_size`: 基础尺寸（默认 1024）
  - 512: 快速模式，64 个视觉 tokens
  - 640: 小尺寸，100 个视觉 tokens
  - 1024: 标准尺寸，256 个视觉 tokens
  - 1280: 大尺寸，400 个视觉 tokens

- `image_size`: 裁剪尺寸（默认 640）
  - 用于动态分辨率模式

### 显存占用

- 512×512: ~8GB
- 640×640: ~10GB
- 1024×1024: ~14GB
- 1280×1280: ~18GB

---

## ❓ 常见问题

### 1. 模型下载慢

设置 Hugging Face 镜像：

```bash
set HF_ENDPOINT=https://hf-mirror.com
```

### 2. CUDA 内存不足

- 减小 `base_size` 参数
- 关闭其他占用 GPU 的程序
- 使用更小的批处理大小

### 3. Flash Attention 安装失败

可以跳过，修改 `api_server.py`：

```python
model = AutoModel.from_pretrained(
    MODEL_NAME,
    _attn_implementation='eager',  # 改为 eager
    trust_remote_code=True,
    use_safetensors=True
)
```

### 4. 端口被占用

修改 `api_server.py` 中的端口：

```python
uvicorn.run(
    "api_server:app",
    host="0.0.0.0",
    port=8001,  # 改为其他端口
    reload=False,
    workers=1
)
```

---

## 📁 项目结构

```
deepseek-OCR/
├── api_server.py           # FastAPI 服务主文件
├── test_api.py             # API 测试脚本
├── Dockerfile              # Docker 镜像配置
├── docker-compose.yml      # Docker Compose 配置
├── install.bat             # Windows 安装脚本
├── start_server.bat        # Windows 启动脚本
├── requirements.txt        # Python 依赖
├── DEPLOYMENT.md           # 详细部署文档
├── outputs/                # 输出目录
├── models/                 # 模型缓存目录
└── DeepSeek-OCR-master/    # 原始代码
```

---

## 🚀 性能优化建议

1. **使用批量接口**: 一次处理多张图片
2. **启用模型缓存**: 避免重复下载模型
3. **调整图片尺寸**: 根据需求平衡速度和精度
4. **使用 GPU**: 确保 CUDA 可用
5. **并发处理**: 使用多线程处理多个请求

---

## 📞 技术支持

- 项目地址: https://github.com/deepseek-ai/DeepSeek-OCR
- 问题反馈: 提交 GitHub Issue
- 详细文档: 查看 DEPLOYMENT.md

---

## 📄 许可证

MIT License

