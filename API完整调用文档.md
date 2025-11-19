# DeepSeek-OCR API 完整调用文档

## 📋 概述

DeepSeek-OCR API 提供 4 个主要端点，支持**图片和 PDF 文件**的 OCR 识别：

1. **`POST /ocr/image`** - 纯文本 OCR（图片或 PDF）
2. **`POST /ocr/batch`** - 批量 OCR（多个图片或 PDF）
3. **`POST /ocr/image/boxes`** - 画框 API（图片或 PDF，返回边界框）
4. **`POST /ocr/image/extract`** - 提取 API（图片或 PDF，返回子图）

### 支持的文件格式

- **图片**: JPG, PNG, JPEG, BMP, GIF, TIFF, WEBP
- **PDF**: 自动识别并处理所有页面

### ⚠️ 重要提示

**客户端超时设置：**
- **推荐超时时间：300 秒（5 分钟）**
- 首次请求需要加载模型（约 27 秒）
- PDF 多页处理需要更长时间
- **必须添加 `Connection: close` 头部**（Windows Docker 环境）

---

## 1️⃣ 纯文本 OCR - `/ocr/image`

### 功能说明

对图片或 PDF 进行 OCR 识别，只返回文本内容。

### HTTP 请求

**端点**: `POST http://localhost:8200/ocr/image`

**Content-Type**: `multipart/form-data`

**完整请求头**:
```json
{
  "Connection": "close",
  "Content-Type": "multipart/form-data"
}
```

**请求参数**:

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `file` | File | ✅ | - | 图片或 PDF 文件 |
| `prompt` | String | ❌ | `<image>\n<\|grounding\|>Convert the document to markdown.` | OCR 提示词 |
| `base_size` | Integer | ❌ | 1024 | 基础尺寸 |
| `image_size` | Integer | ❌ | 640 | 图片尺寸 |
| `crop_mode` | Boolean | ❌ | true | 是否裁剪模式 |
| `save_results` | Boolean | ❌ | false | 是否保存结果到服务器 |
| `test_compress` | Boolean | ❌ | false | 是否测试压缩 |

**完整请求参数（Form Data）**:
```json
{
  "file": "<binary file data>",
  "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
  "base_size": "1024",
  "image_size": "640",
  "crop_mode": "true",
  "save_results": "false",
  "test_compress": "false"
}
```

### 响应格式

```json
{
  "success": true,
  "text": "识别的文本内容...",
  "processing_time": 15.8,
  "metadata": {
    "filename": "document.pdf",
    "file_size": 327680,
    "file_type": "pdf",
    "total_pages": 3,
    "base_size": 1024,
    "image_size": 640,
    "crop_mode": true
  },
  "image_with_boxes_url": null,
  "extracted_images_urls": null
}
```

### 调用示例

#### Python (requests)

```python
import requests

# 重要：必须添加 Connection: close 头部
session = requests.Session()
session.headers.update({'Connection': 'close'})

# 上传图片
with open('document.png', 'rb') as f:
    files = {'file': ('document.png', f, 'image/png')}
    data = {
        'prompt': '<image>\n<|grounding|>Convert the document to markdown.',
        'base_size': 1024,
        'image_size': 640
    }
    
    response = session.post(
        'http://localhost:8200/ocr/image',
        files=files,
        data=data,
        timeout=300  # 5 分钟超时
    )

result = response.json()
print(f"识别文本: {result['text']}")
```

#### Python (上传 PDF)

```python
import requests

session = requests.Session()
session.headers.update({'Connection': 'close'})

# 上传 PDF
with open('document.pdf', 'rb') as f:
    files = {'file': ('document.pdf', f, 'application/pdf')}
    data = {'prompt': '<image>\n<|grounding|>Convert the document to markdown.'}
    
    response = session.post(
        'http://localhost:8200/ocr/image',
        files=files,
        data=data,
        timeout=300
    )

result = response.json()
print(f"PDF 页数: {result['metadata']['total_pages']}")
print(f"识别文本: {result['text']}")
```

#### JavaScript (fetch)

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);  // 可以是图片或 PDF
formData.append('prompt', '<image>\\n<|grounding|>Convert the document to markdown.');

const controller = new AbortController();
const timeoutId = setTimeout(() => controller.abort(), 300000);  // 5 分钟超时

try {
  const response = await fetch('http://localhost:8200/ocr/image', {
    method: 'POST',
    headers: {
      'Connection': 'close'  // 重要！
    },
    body: formData,
    signal: controller.signal
  });
  
  clearTimeout(timeoutId);
  const result = await response.json();
  console.log('识别文本:', result.text);
  
  if (result.metadata.file_type === 'pdf') {
    console.log('PDF 页数:', result.metadata.total_pages);
  }
} catch (error) {
  console.error('请求失败:', error);
}
```

#### cURL

```bash
curl -X POST http://localhost:8200/ocr/image \
  -H "Connection: close" \
  -F "file=@document.png" \
  -F "prompt=<image>\n<|grounding|>Convert the document to markdown." \
  -F "base_size=1024" \
  -F "image_size=640" \
  --max-time 300
```

---

## 2️⃣ 批量 OCR - `/ocr/batch`

### 功能说明

批量处理多个图片或 PDF 文件。

### HTTP 请求

**端点**: `POST http://localhost:8200/ocr/batch`

**Content-Type**: `multipart/form-data`

**完整请求头**:
```json
{
  "Connection": "close",
  "Content-Type": "multipart/form-data"
}
```

**请求参数**:

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `files` | File[] | ✅ | - | 多个图片或 PDF 文件 |
| `prompt` | String | ❌ | `<image>\n<\|grounding\|>Convert the document to markdown.` | OCR 提示词 |
| 其他参数 | - | ❌ | - | 同 `/ocr/image` |

**完整请求参数（Form Data）**:
```json
{
  "files": ["<binary file data 1>", "<binary file data 2>", "..."],
  "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
  "base_size": "1024",
  "image_size": "640",
  "crop_mode": "true"
}
```

### 响应格式

```json
[
  {
    "success": true,
    "text": "第一个文件的文本...",
    "processing_time": 15.8,
    "metadata": { ... }
  },
  {
    "success": true,
    "text": "第二个文件的文本...",
    "processing_time": 12.3,
    "metadata": { ... }
  }
]
```

### 调用示例

#### Python

```python
import requests

session = requests.Session()
session.headers.update({'Connection': 'close'})

files = [
    ('files', ('doc1.png', open('doc1.png', 'rb'), 'image/png')),
    ('files', ('doc2.pdf', open('doc2.pdf', 'rb'), 'application/pdf'))
]

data = {'prompt': '<image>\n<|grounding|>Convert the document to markdown.'}

response = session.post(
    'http://localhost:8200/ocr/batch',
    files=files,
    data=data,
    timeout=600  # 批量处理需要更长时间
)

results = response.json()
for i, result in enumerate(results):
    print(f"文件 {i+1}: {result['text'][:100]}...")
```

---

## 3️⃣ 画框 API - `/ocr/image/boxes`

### 功能说明

对图片或 PDF 进行 OCR 识别，在图片上画边界框，返回结构化框信息。

### HTTP 请求

**端点**: `POST http://localhost:8200/ocr/image/boxes`

**Content-Type**: `multipart/form-data`

**完整请求头**:
```json
{
  "Connection": "close",
  "Content-Type": "multipart/form-data"
}
```

**请求参数**:

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `file` | File | ✅ | - | 图片或 PDF 文件 |
| `prompt` | String | ❌ | `<image>\n<\|grounding\|>Convert the document to markdown.` | OCR 提示词 |
| `include_text` | Boolean | ❌ | true | 是否返回完整 OCR 文本 |
| 其他参数 | - | ❌ | - | 同 `/ocr/image` |

**完整请求参数（Form Data）**:
```json
{
  "file": "<binary file data>",
  "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
  "include_text": "true",
  "base_size": "1024",
  "image_size": "640",
  "crop_mode": "true"
}
```

### 响应格式

#### 图片文件响应

```json
{
  "success": true,
  "image_with_boxes_url": "http://localhost:8200/outputs/image_with_boxes_20231119_143052.png",
  "boxes": [
    {
      "id": "box_1",
      "label_type": "title",
      "x1": 0.1, "y1": 0.05, "x2": 0.9, "y2": 0.15,
      "x1_px": 100, "y1_px": 50, "x2_px": 900, "y2_px": 150
    },
    {
      "id": "box_2",
      "label_type": "paragraph",
      "x1": 0.1, "y1": 0.2, "x2": 0.9, "y2": 0.5,
      "x1_px": 100, "y1_px": 200, "x2_px": 900, "y2_px": 500
    }
  ],
  "text": "完整的 OCR 文本...",
  "question": "<image>\n<|grounding|>Convert the document to markdown.",
  "labels_summary": ["title", "paragraph", "image"],
  "processing_time": 18.5,
  "metadata": {
    "filename": "document.png",
    "file_size": 327680,
    "file_type": "image",
    "boxes_count": 2
  }
}
```

#### PDF 文件响应

```json
{
  "success": true,
  "image_with_boxes_url": "http://localhost:8200/outputs/document_page1_boxes.jpg",
  "boxes": [
    {
      "id": "box_1",
      "label_type": "title",
      "x1": 0.1, "y1": 0.05, "x2": 0.9, "y2": 0.15,
      "x1_px": 100, "y1_px": 50, "x2_px": 900, "y2_px": 150
    }
  ],
  "text": "# 第 1 页\n\n完整的 OCR 文本...\n\n---\n\n# 第 2 页\n\n...",
  "question": "<image>\n<|grounding|>Convert the document to markdown.",
  "labels_summary": ["title", "paragraph", "image"],
  "processing_time": 85.3,
  "metadata": {
    "filename": "document.pdf",
    "file_size": 1048576,
    "file_type": "pdf",
    "total_pages": 5,
    "boxes_count": 58,
    "all_pages_urls": [
      "http://localhost:8200/outputs/document_page1_boxes.jpg",
      "http://localhost:8200/outputs/document_page2_boxes.jpg",
      "http://localhost:8200/outputs/document_page3_boxes.jpg",
      "http://localhost:8200/outputs/document_page4_boxes.jpg",
      "http://localhost:8200/outputs/document_page5_boxes.jpg"
    ]
  }
}
```

**PDF 特殊说明**：
- `image_with_boxes_url` 返回第一页的画框图片
- `metadata.all_pages_urls` 包含所有页的画框图片 URL 列表
- `boxes` 包含所有页的边界框信息
- `text` 包含所有页的文本，用 `---` 分隔，每页标注 `# 第 X 页`

### 调用示例

#### Python

```python
import requests

session = requests.Session()
session.headers.update({'Connection': 'close'})

with open('document.png', 'rb') as f:
    files = {'file': ('document.png', f, 'image/png')}
    data = {
        'prompt': '<image>\n<|grounding|>Convert the document to markdown.',
        'include_text': 'true'
    }

    response = session.post(
        'http://localhost:8200/ocr/image/boxes',
        files=files,
        data=data,
        timeout=300
    )

result = response.json()
print(f"画框图片 URL: {result['image_with_boxes_url']}")
print(f"检测到 {len(result['boxes'])} 个区域")
print(f"标签类型: {result['labels_summary']}")

# 遍历所有边界框
for box in result['boxes']:
    print(f"区域 {box['id']}: {box['label_type']} - 坐标 ({box['x1_px']}, {box['y1_px']}) 到 ({box['x2_px']}, {box['y2_px']})")
```

#### JavaScript

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('prompt', '<image>\\n<|grounding|>Convert the document to markdown.');
formData.append('include_text', 'true');

const response = await fetch('http://localhost:8200/ocr/image/boxes', {
  method: 'POST',
  headers: { 'Connection': 'close' },
  body: formData,
  signal: AbortSignal.timeout(300000)
});

const result = await response.json();
console.log('画框图片:', result.image_with_boxes_url);
console.log('边界框数量:', result.boxes.length);
console.log('标签类型:', result.labels_summary);

// 在前端显示画框图片
document.getElementById('result-image').src = result.image_with_boxes_url;
```

---

## 4️⃣ 提取 API - `/ocr/image/extract`

### 功能说明

对图片或 PDF 进行 OCR 识别，提取图片区域（如文档中的图表），返回子图和对应文字。

### HTTP 请求

**端点**: `POST http://localhost:8200/ocr/image/extract`

**Content-Type**: `multipart/form-data`

**完整请求头**:
```json
{
  "Connection": "close",
  "Content-Type": "multipart/form-data"
}
```

**请求参数**:

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `file` | File | ✅ | - | 图片或 PDF 文件 |
| `prompt` | String | ❌ | `<image>\n<\|grounding\|>Convert the document to markdown.` | OCR 提示词 |
| `include_text` | Boolean | ❌ | true | 是否返回完整 OCR 文本 |
| `include_boxes` | Boolean | ❌ | true | 是否返回边界框信息 |
| 其他参数 | - | ❌ | - | 同 `/ocr/image` |

**完整请求参数（Form Data）**:
```json
{
  "file": "<binary file data>",
  "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
  "include_text": "true",
  "include_boxes": "true",
  "base_size": "1024",
  "image_size": "640",
  "crop_mode": "true"
}
```

### 响应格式

```json
{
  "success": true,
  "text": "完整的 OCR 文本...",
  "regions": [
    {
      "id": "region_1",
      "label_type": "image",
      "bbox": {
        "id": "box_1",
        "label_type": "image",
        "x1": 0.2, "y1": 0.3, "x2": 0.8, "y2": 0.7,
        "x1_px": 200, "y1_px": 300, "x2_px": 800, "y2_px": 700
      },
      "image_url": "http://localhost:8200/outputs/extracted_image_1_20231119_143052.png",
      "text": "这个区域的文字说明..."
    }
  ],
  "question": "<image>\n<|grounding|>Convert the document to markdown.",
  "processing_time": 22.3,
  "metadata": { ... }
}
```

### 调用示例

#### Python

```python
import requests

session = requests.Session()
session.headers.update({'Connection': 'close'})

with open('document.png', 'rb') as f:
    files = {'file': ('document.png', f, 'image/png')}
    data = {
        'prompt': '<image>\n<|grounding|>Convert the document to markdown.',
        'include_text': 'true',
        'include_boxes': 'true'
    }

    response = session.post(
        'http://localhost:8200/ocr/image/extract',
        files=files,
        data=data,
        timeout=300
    )

result = response.json()
print(f"提取了 {len(result['regions'])} 个区域")

# 遍历所有提取的区域
for region in result['regions']:
    print(f"\n区域 {region['id']}:")
    print(f"  类型: {region['label_type']}")
    print(f"  图片 URL: {region['image_url']}")
    print(f"  文字: {region['text'][:100]}...")

    # 下载子图
    img_response = requests.get(region['image_url'])
    with open(f"{region['id']}.png", 'wb') as img_file:
        img_file.write(img_response.content)
```

#### JavaScript

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('prompt', '<image>\\n<|grounding|>Convert the document to markdown.');
formData.append('include_text', 'true');
formData.append('include_boxes', 'true');

const response = await fetch('http://localhost:8200/ocr/image/extract', {
  method: 'POST',
  headers: { 'Connection': 'close' },
  body: formData,
  signal: AbortSignal.timeout(300000)
});

const result = await response.json();
console.log('提取区域数:', result.regions.length);

// 显示所有提取的子图
result.regions.forEach((region, index) => {
  const img = document.createElement('img');
  img.src = region.image_url;
  img.alt = `${region.label_type} - ${region.text.substring(0, 50)}`;
  document.getElementById('extracted-images').appendChild(img);
});
```

---

## 📝 常见问题

### 1. 请求超时怎么办？

**问题**: 客户端一直等待，没有收到响应。

**解决方案**:
1. **增加超时时间**: 设置为 300 秒（5 分钟）或更长
2. **添加 Connection: close 头部**: 在 Windows Docker 环境下必须添加
3. **检查文件大小**: 确保文件小于 20MB

### 2. PDF 处理很慢怎么办？

**问题**: PDF 文件有很多页，处理时间很长。

**解决方案**:
1. **拆分 PDF**: 将大 PDF 拆分为多个小 PDF
2. **使用批量 API**: 并行处理多个小 PDF
3. **增加超时时间**: 根据页数调整超时时间（每页约 10-20 秒）

### 3. 如何判断上传的是图片还是 PDF？

**答案**: 不需要判断！API 会自动识别文件类型。前端只需要上传文件即可。

### 4. 响应中的 URL 无法访问怎么办？

**问题**: `image_with_boxes_url` 或 `image_url` 返回的 URL 无法访问。

**解决方案**:
1. 检查 `API_BASE_URL` 环境变量是否正确设置
2. 如果使用内网穿透，需要设置为穿透后的地址
3. 确保 `/outputs` 目录可以通过 HTTP 访问

---

## 🔧 环境变量配置

在 `docker-compose.yml` 中可以配置以下环境变量：

```yaml
environment:
  - API_BASE_URL=http://localhost:8200  # API 基础 URL
  - MAX_FILE_SIZE=20971520              # 最大文件大小（20MB）
  - MAX_CONCURRENT_REQUESTS=1           # 最大并发请求数
  - IDLE_TIMEOUT=3600                   # 空闲超时（秒）
  - LAZY_LOAD=true                      # 按需加载模型
```

---

## 📞 技术支持

如有问题，请查看：
- Docker 日志: `docker logs deepseek-ocr-service`
- API 文档: `http://localhost:8200/docs`
- 健康检查: `http://localhost:8200/health`


