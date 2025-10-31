# DeepSeek-OCR API 调用规则

## 📋 目录

- [服务概述](#服务概述)
- [API 端点](#api-端点)
- [请求参数](#请求参数)
- [响应格式](#响应格式)
- [使用示例](#使用示例)
- [性能指标](#性能指标)
- [注意事项](#注意事项)
- [故障排查](#故障排查)

---

## 服务概述

DeepSeek-OCR 是一个基于深度学习的 OCR (光学字符识别) 服务,支持:
- ✅ 中英文混合识别
- ✅ 文档转 Markdown 格式
- ✅ 文本位置定位 (Grounding)
- ✅ GPU 加速推理
- ✅ 批量处理
- ✅ Base64 图片输入

**服务地址**: `http://localhost:8200`

**技术栈**:
- 模型: DeepSeek-OCR (deepseek-ai/DeepSeek-OCR)
- 框架: PyTorch 2.10.0 (nightly) + CUDA 12.8
- API: FastAPI
- GPU: NVIDIA RTX 5060 Ti (16GB)

---

## API 端点

### 1. 健康检查

**端点**: `GET /health`

**描述**: 检查服务状态和模型加载情况

**响应示例**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "gpu_available": true,
  "timestamp": "2025-10-30T11:30:33.692626"
}
```

---

### 2. 单图片 OCR 识别

**端点**: `POST /ocr/image`

**描述**: 对单张图片进行 OCR 识别

**请求参数**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `file` | File | ✅ | - | 图片文件 (支持 jpg, png, bmp, gif, tiff, webp) |
| `prompt` | String | ❌ | `<image>\n<|grounding|>Convert the document to markdown.` | OCR 提示词,必须包含 `<image>` 标签 |
| `base_size` | Integer | ❌ | 1024 | 基础图片尺寸 |
| `image_size` | Integer | ❌ | 640 | 处理图片尺寸 |
| `crop_mode` | Boolean | ❌ | true | 是否裁剪图片 |
| `save_results` | Boolean | ❌ | false | 是否保存结果文件 |
| `test_compress` | Boolean | ❌ | false | 是否测试压缩 (建议保持 false) |

**响应格式**:
```json
{
  "success": true,
  "text": "识别的文本内容...",
  "processing_time": 2.73,
  "error": null,
  "metadata": {
    "filename": "test_image.jpg",
    "file_size": 35561,
    "base_size": 1024,
    "image_size": 640,
    "crop_mode": true
  }
}
```

---

### 3. 批量 OCR 识别

**端点**: `POST /ocr/batch`

**描述**: 对多张图片进行批量 OCR 识别

**请求参数**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `files` | File[] | ✅ | - | 多个图片文件 |
| `prompt` | String | ❌ | `<image>\n<|grounding|>Convert the document to markdown.` | OCR 提示词 |
| `base_size` | Integer | ❌ | 1024 | 基础图片尺寸 |
| `image_size` | Integer | ❌ | 640 | 处理图片尺寸 |
| `crop_mode` | Boolean | ❌ | true | 是否裁剪图片 |

**响应格式**: 返回数组,每个元素对应一张图片的识别结果

---

### 4. Base64 图片 OCR 识别

**端点**: `POST /ocr/base64`

**描述**: 对 Base64 编码的图片进行 OCR 识别

**请求参数**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `image_base64` | String | ✅ | - | Base64 编码的图片数据 |
| `prompt` | String | ❌ | `<image>\n<|grounding|>Convert the document to markdown.` | OCR 提示词 |
| `base_size` | Integer | ❌ | 1024 | 基础图片尺寸 |
| `image_size` | Integer | ❌ | 640 | 处理图片尺寸 |
| `crop_mode` | Boolean | ❌ | true | 是否裁剪图片 |
| `save_results` | Boolean | ❌ | false | 是否保存结果文件 |

---

## 使用示例

### Python 示例

#### 1. 健康检查
```python
import requests

response = requests.get("http://localhost:8200/health")
print(response.json())
```

#### 2. 单图片 OCR
```python
import requests

# 方式 1: 使用默认参数
with open("image.jpg", "rb") as f:
    files = {"file": ("image.jpg", f, "image/jpeg")}
    response = requests.post("http://localhost:8200/ocr/image", files=files)
    result = response.json()
    print(result["text"])

# 方式 2: 自定义参数
with open("image.jpg", "rb") as f:
    files = {"file": ("image.jpg", f, "image/jpeg")}
    data = {
        "prompt": "<image>\n<|grounding|>Extract all text from the image.",
        "base_size": 1024,
        "image_size": 640,
        "crop_mode": True
    }
    response = requests.post("http://localhost:8200/ocr/image", files=files, data=data)
    result = response.json()
    print(result["text"])
```

#### 3. 批量 OCR
```python
import requests

files = [
    ("files", ("image1.jpg", open("image1.jpg", "rb"), "image/jpeg")),
    ("files", ("image2.jpg", open("image2.jpg", "rb"), "image/jpeg")),
]

response = requests.post("http://localhost:8200/ocr/batch", files=files)
results = response.json()

for i, result in enumerate(results):
    print(f"Image {i+1}: {result['text']}")
```

#### 4. Base64 图片 OCR
```python
import requests
import base64

# 读取图片并转换为 Base64
with open("image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

# 发送请求
response = requests.post(
    "http://localhost:8200/ocr/base64",
    json={"image_base64": image_base64}
)
result = response.json()
print(result["text"])
```

---

### cURL 示例

#### 1. 健康检查
```bash
curl http://localhost:8200/health
```

#### 2. 单图片 OCR
```bash
curl -X POST http://localhost:8200/ocr/image \
  -F "file=@image.jpg"
```

#### 3. 自定义参数
```bash
curl -X POST http://localhost:8200/ocr/image \
  -F "file=@image.jpg" \
  -F "prompt=<image>\n<|grounding|>Extract all text." \
  -F "base_size=1024" \
  -F "image_size=640" \
  -F "crop_mode=true"
```

#### 4. 批量 OCR
```bash
curl -X POST http://localhost:8200/ocr/batch \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"
```

#### 5. Base64 图片 OCR
```bash
# 先将图片转换为 Base64
IMAGE_BASE64=$(base64 -w 0 image.jpg)

# 发送请求
curl -X POST http://localhost:8200/ocr/base64 \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"$IMAGE_BASE64\"}"
```

---

### HTTP 请求/响应格式 (JSON)

#### 1. 健康检查

**请求**:
```
GET http://localhost:8200/health
```

**响应 JSON**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "gpu_available": true,
  "timestamp": "2025-10-30T11:30:33.692626"
}
```

---

#### 2. 单图片 OCR

**请求格式**:
```
POST http://localhost:8200/ocr/image
Content-Type: multipart/form-data
```

**请求参数 (form-data)**:
```json
{
  "file": "<图片文件二进制数据>",
  "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
  "base_size": 1024,
  "image_size": 640,
  "crop_mode": true,
  "save_results": false,
  "test_compress": false
}
```

**参数说明**:
- `file`: 图片文件 (必需) - 二进制文件数据
- `prompt`: 提示词 (可选，默认: `"<image>\n<|grounding|>Convert the document to markdown."`)
- `base_size`: 基础尺寸 (可选，默认: `1024`)
- `image_size`: 处理尺寸 (可选，默认: `640`)
- `crop_mode`: 是否裁剪 (可选，默认: `true`)
- `save_results`: 是否保存结果 (可选，默认: `false`)
- `test_compress`: 是否测试压缩 (可选，默认: `false`)

**响应 JSON**:
```json
{
  "success": true,
  "text": "# DeepSeek-OCR Test\n\nThis is a test document for OCR.\n\n## Features\n- High accuracy\n- Fast processing\n- GPU acceleration\n\nDate: 2025-10-30",
  "processing_time": 2.73,
  "error": null,
  "metadata": {
    "filename": "image.jpg",
    "file_size": 35561,
    "base_size": 1024,
    "image_size": 640,
    "crop_mode": true
  }
}
```

---

#### 3. 批量 OCR

**请求格式**:
```
POST http://localhost:8200/ocr/batch
Content-Type: multipart/form-data
```

**请求参数 (form-data)**:
```json
{
  "files": ["<图片文件1二进制数据>", "<图片文件2二进制数据>", "<图片文件3二进制数据>"],
  "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
  "base_size": 1024,
  "image_size": 640,
  "crop_mode": true
}
```

**参数说明**:
- `files`: 多个图片文件 (必需) - 数组形式的二进制文件数据
- `prompt`: 提示词 (可选)
- `base_size`: 基础尺寸 (可选)
- `image_size`: 处理尺寸 (可选)
- `crop_mode`: 是否裁剪 (可选)

**响应 JSON** (数组):
```json
[
  {
    "success": true,
    "text": "第一张图片的文本内容...",
    "processing_time": 2.73,
    "error": null,
    "metadata": {
      "filename": "image1.jpg",
      "file_size": 35561,
      "base_size": 1024,
      "image_size": 640,
      "crop_mode": true
    }
  },
  {
    "success": true,
    "text": "第二张图片的文本内容...",
    "processing_time": 2.85,
    "error": null,
    "metadata": {
      "filename": "image2.jpg",
      "file_size": 42103,
      "base_size": 1024,
      "image_size": 640,
      "crop_mode": true
    }
  }
]
```

---

#### 4. Base64 图片 OCR

**请求格式**:
```
POST http://localhost:8200/ocr/base64
Content-Type: application/json
```

**请求 JSON**:
```json
{
  "image_base64": "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCABgAGADASIA...",
  "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
  "base_size": 1024,
  "image_size": 640,
  "crop_mode": true,
  "save_results": false
}
```

**参数说明**:
- `image_base64`: Base64 编码的图片数据 (必需)
- `prompt`: 提示词 (可选，默认: `"<image>\n<|grounding|>Convert the document to markdown."`)
- `base_size`: 基础尺寸 (可选，默认: `1024`)
- `image_size`: 处理尺寸 (可选，默认: `640`)
- `crop_mode`: 是否裁剪 (可选，默认: `true`)
- `save_results`: 是否保存结果 (可选，默认: `false`)

**响应 JSON**:
```json
{
  "success": true,
  "text": "识别的文本内容...",
  "processing_time": 2.73,
  "error": null,
  "metadata": {
    "filename": "base64_image.jpg",
    "file_size": 35561,
    "base_size": 1024,
    "image_size": 640,
    "crop_mode": true
  }
}
```

---

#### 5. 错误响应

**请求错误 (400)**:
```json
{
  "detail": "只支持图片文件"
}
```

**处理失败 (200, success=false)**:
```json
{
  "success": false,
  "text": null,
  "processing_time": null,
  "error": "处理图片时发生错误: ...",
  "metadata": null
}
```

**服务器错误 (500)**:
```json
{
  "detail": "Internal server error"
}
```

---

### JavaScript 示例

#### 1. 使用 Fetch API
```javascript
// 单图片 OCR
async function ocrImage(file) {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('http://localhost:8200/ocr/image', {
    method: 'POST',
    body: formData
  });
  
  const result = await response.json();
  console.log(result.text);
}

// 使用示例
const fileInput = document.querySelector('input[type="file"]');
fileInput.addEventListener('change', (e) => {
  ocrImage(e.target.files[0]);
});
```

#### 2. Base64 图片 OCR
```javascript
async function ocrBase64(imageBase64) {
  const response = await fetch('http://localhost:8200/ocr/base64', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ image_base64: imageBase64 })
  });
  
  const result = await response.json();
  console.log(result.text);
}
```

---

## 性能指标

### 硬件配置
- **GPU**: NVIDIA GeForce RTX 5060 Ti (16GB VRAM)
- **CPU**: 64GB RAM
- **CUDA**: 12.9
- **PyTorch**: 2.10.0.dev20251029+cu128

### 性能数据
- **模型加载时间**: ~35 秒
- **GPU 显存占用**: 7.6GB - 9.6GB
- **单图片推理时间**: ~2.7 秒
- **API 响应时间**: ~6.9 秒 (包含网络和文件处理)
- **GPU 利用率**: 2% (空闲) → 80-90% (推理中)

### 并发性能
- **推荐并发数**: 1-2 (单 GPU)
- **最大并发数**: 取决于显存大小

---

## 注意事项

### 1. Prompt 格式要求
⚠️ **重要**: `prompt` 参数必须包含 `<image>` 标签,否则无法返回识别结果!

**正确示例**:
```
<image>\n<|grounding|>Convert the document to markdown.
<image>\n<|grounding|>Extract all text from the image.
<image>\nRecognize the text in this image.
```

**错误示例**:
```
Convert the document to markdown.  ❌ (缺少 <image> 标签)
Extract all text.  ❌ (缺少 <image> 标签)
```

### 2. 图片格式支持
支持的格式: `jpg`, `jpeg`, `png`, `bmp`, `gif`, `tiff`, `webp`

### 3. 图片大小限制
- 建议图片大小: < 10MB
- 最大图片尺寸: 取决于 GPU 显存

### 4. test_compress 参数
⚠️ 建议保持 `test_compress=false`,避免干扰 `eval_mode` 的返回逻辑

### 5. GPU 兼容性
- 需要 NVIDIA GPU 支持 CUDA 12.x
- RTX 50 系列 (sm_120) 需要 PyTorch nightly 版本

---

## 故障排查

### 1. 服务无法启动

**问题**: 容器启动失败或一直重启

**解决方案**:
```bash
# 查看容器日志
docker logs deepseek-ocr-service --tail 100

# 检查 GPU 是否可用
nvidia-smi

# 重启容器
docker-compose down
docker-compose up -d
```

### 2. OCR 返回 null

**问题**: API 返回 `"text": null`

**原因**: 
- Prompt 缺少 `<image>` 标签
- `test_compress=true` 干扰了返回逻辑

**解决方案**:
- 确保 prompt 包含 `<image>` 标签
- 设置 `test_compress=false`

### 3. CUDA 错误

**问题**: `CUDA error: no kernel image is available`

**原因**: PyTorch 版本不支持当前 GPU 架构

**解决方案**:
- RTX 50 系列需要 PyTorch nightly + CUDA 12.8
- 参考 Dockerfile 中的安装命令

### 4. 显存不足

**问题**: `CUDA out of memory`

**解决方案**:
- 减少并发请求数
- 降低 `base_size` 和 `image_size` 参数
- 增加 GPU 显存或使用更小的模型

### 5. 处理速度慢

**问题**: OCR 处理时间过长

**检查项**:
- GPU 是否正常工作: `nvidia-smi`
- 是否使用了 GPU 模式: 检查 `/health` 端点的 `device` 字段
- 是否有其他进程占用 GPU

---

## 附录

### Docker 部署命令

```bash
# 构建镜像
docker-compose build

# 启动服务
docker-compose up -d

# 查看日志
docker logs deepseek-ocr-service -f

# 停止服务
docker-compose down

# 重启服务
docker-compose restart
```

### 环境变量

可以在 `docker-compose.yml` 中配置以下环境变量:

```yaml
environment:
  - MODEL_NAME=deepseek-ai/DeepSeek-OCR
  - OUTPUT_DIR=/app/outputs
  - HF_HOME=/app/models
```

---

## 联系方式

如有问题,请联系技术支持或查看项目文档:
- GitHub: https://github.com/deepseek-ai/DeepSeek-OCR
- 文档: 本地部署文档

---

**最后更新**: 2025-10-30
**版本**: 1.0.0

