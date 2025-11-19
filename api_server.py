"""
DeepSeek-OCR FastAPI Server
提供 RESTful API 接口用于图片和 PDF 的 OCR 处理
"""

import os
# 禁用 torch.compile 和 JIT 编译,避免 CUDA 架构不兼容问题
os.environ["PYTORCH_JIT"] = "0"
os.environ["PYTORCH_NVFUSER_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import torch
import os
import tempfile
import base64
from pathlib import Path
import logging
from datetime import datetime, timedelta
import uvicorn
import asyncio
import gc
import threading
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import fitz  # PyMuPDF - 用于 PDF 处理

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="DeepSeek-OCR API",
    description="基于 DeepSeek-OCR 的光学字符识别服务",
    version="1.0.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量存储模型
model = None
tokenizer = None
MODEL_LOADED = False
last_request_time = None
model_load_lock = threading.Lock()

# ==================== 配置 ====================
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-OCR")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 输出目录配置
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs"))
OUTPUT_DIR.mkdir(exist_ok=True)

# 输出文件保留时间（秒）- 默认 24 小时
OUTPUT_FILE_RETENTION = int(os.getenv("OUTPUT_FILE_RETENTION", str(24 * 3600)))

# 空闲超时配置（秒）
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", "3600"))  # 默认 1 小时
LAZY_LOAD = os.getenv("LAZY_LOAD", "true").lower() == "true"  # 是否启用按需加载

# ==================== 并发控制配置 ====================
# 最大并发请求数（防止 GPU 溢出）
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "1"))
# 最大文件大小（字节）- 默认 20MB
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(20 * 1024 * 1024)))
# 并发控制信号量
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
# 当前处理中的请求数
active_requests = 0
active_requests_lock = threading.Lock()

# ==================== API 基础 URL 配置 ====================
# 用于生成完整的文件访问 URL
# 如果部署在公网，设置为公网地址，例如: "https://your-domain.com"
# 如果使用内网穿透，设置为穿透后的地址，例如: "https://your-tunnel.com"
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8200")


class OCRRequest(BaseModel):
    """OCR 请求模型"""
    prompt: Optional[str] = "<image>\n<|grounding|>Convert the document to markdown."
    base_size: Optional[int] = 1024
    image_size: Optional[int] = 640
    crop_mode: Optional[bool] = True
    save_results: Optional[bool] = False
    test_compress: Optional[bool] = True


class OCRResponse(BaseModel):
    """OCR 响应模型（纯文本版，保留向后兼容）"""
    success: bool
    text: Optional[str] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
    metadata: Optional[dict] = None
    image_with_boxes_url: Optional[str] = None  # 已废弃，始终为 None
    extracted_images_urls: Optional[List[str]] = None  # 已废弃，始终为 None


class BoundingBox(BaseModel):
    """边界框信息"""
    id: str  # 框的唯一 ID
    label_type: str  # 标签类型：image, title, paragraph, table 等
    # 归一化坐标 (0-1)
    x1: float
    y1: float
    x2: float
    y2: float
    # 像素坐标
    x1_px: int
    y1_px: int
    x2_px: int
    y2_px: int


class Region(BaseModel):
    """提取的图片区域"""
    id: str  # 区域 ID
    label_type: str  # 标签类型
    page_number: Optional[int] = None  # 页码（PDF 时有效）
    bbox: Optional[BoundingBox] = None  # 边界框信息（可选）
    image_url: str  # 裁剪后的子图 URL
    text: Optional[str] = None  # 该区域对应的文字内容


class OCRBoxesResponse(BaseModel):
    """画框 API 响应模型"""
    success: bool
    image_with_boxes_url: str  # 画好框的整图 URL
    boxes: List[BoundingBox]  # 所有框的结构化信息
    text: Optional[str] = None  # 整页 OCR 文本（可选）
    question: str  # 回显 prompt
    labels_summary: List[str]  # 检测到的标签类型列表
    processing_time: float
    metadata: dict
    error: Optional[str] = None


class OCRExtractResponse(BaseModel):
    """提取 API 响应模型"""
    success: bool
    text: Optional[str] = None  # 整页 OCR 文本（可选）
    regions: List[Region]  # 提取的图片区域列表
    question: str  # 回显 prompt
    processing_time: float
    metadata: dict
    error: Optional[str] = None


def re_match(text):
    """提取 OCR 结果中的引用和检测标签"""
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    # 分类：图片类型和其他类型
    matches_image = []
    matches_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            matches_image.append(a_match[0])
        else:
            matches_other.append(a_match[0])

    return matches, matches_image, matches_other


def is_pdf(file_content: bytes) -> bool:
    """检测文件是否为 PDF"""
    return file_content[:4] == b'%PDF'


def pdf_to_images(pdf_content: bytes, dpi: int = 200) -> List[Image.Image]:
    """
    将 PDF 转换为图片列表

    Args:
        pdf_content: PDF 文件内容（字节）
        dpi: 渲染 DPI（默认 200，越高越清晰但越慢）

    Returns:
        图片列表，每个元素是一页的 PIL Image
    """
    try:
        # 从字节流打开 PDF
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        images = []

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]

            # 计算缩放比例（DPI / 72）
            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)

            # 渲染页面为图片
            pix = page.get_pixmap(matrix=mat)

            # 转换为 PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(BytesIO(img_data))
            images.append(img)

            logger.info(f"📄 PDF 第 {page_num + 1}/{len(pdf_document)} 页转换完成")

        pdf_document.close()
        return images

    except Exception as e:
        logger.error(f"❌ PDF 转换失败: {e}")
        raise HTTPException(status_code=400, detail=f"PDF 转换失败: {str(e)}")


def extract_coordinates_and_label(ref_text, image_width, image_height):
    """从引用文本中提取坐标和标签"""
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        logger.error(f"提取坐标失败: {e}")
        return None
    return (label_type, cor_list)


def _parse_boxes_from_text(ocr_text: str, image_width: int, image_height: int) -> List[BoundingBox]:
    """从 OCR 文本中解析所有边界框信息

    Args:
        ocr_text: OCR 识别结果文本
        image_width: 图片宽度（像素）
        image_height: 图片高度（像素）

    Returns:
        BoundingBox 对象列表
    """
    matches, _, _ = re_match(ocr_text)
    boxes = []

    for i, ref in enumerate(matches):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result

                for j, points in enumerate(points_list):
                    x1_norm, y1_norm, x2_norm, y2_norm = points

                    # 转换为像素坐标
                    x1_px = int(x1_norm / 999 * image_width)
                    y1_px = int(y1_norm / 999 * image_height)
                    x2_px = int(x2_norm / 999 * image_width)
                    y2_px = int(y2_norm / 999 * image_height)

                    # 转换为 0-1 归一化坐标
                    x1 = x1_norm / 999
                    y1 = y1_norm / 999
                    x2 = x2_norm / 999
                    y2 = y2_norm / 999

                    box = BoundingBox(
                        id=f"box_{i}_{j}",
                        label_type=label_type,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        x1_px=x1_px,
                        y1_px=y1_px,
                        x2_px=x2_px,
                        y2_px=y2_px
                    )
                    boxes.append(box)
        except Exception as e:
            logger.error(f"解析边界框失败: {e}")
            continue

    return boxes


def _extract_region_text(ocr_text: str, region_index: int) -> Optional[str]:
    """从 OCR 文本中提取特定区域的文字内容

    Args:
        ocr_text: OCR 识别结果文本
        region_index: 区域索引

    Returns:
        该区域对应的文字内容，如果没有则返回 None
    """
    # 简单实现：提取每个 ref/det 块后面的文本，直到下一个 ref 标签
    pattern = r'<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>(.*?)(?=<\|ref\||$)'
    matches = re.findall(pattern, ocr_text, re.DOTALL)

    if region_index < len(matches):
        text_content = matches[region_index][2].strip()
        return text_content if text_content else None

    return None


def draw_bounding_boxes(image: Image.Image, ocr_text: str, extract_images: bool = False,
                       save_to_disk: bool = True, filename_prefix: str = "result") -> tuple:
    """在图片上绘制边界框，并可选提取图片区域

    Args:
        image: PIL Image 对象
        ocr_text: OCR 识别结果文本（包含 <|ref|> 和 <|det|> 标签）
        extract_images: 是否提取图片区域
        save_to_disk: 是否保存到磁盘
        filename_prefix: 文件名前缀

    Returns:
        (带边界框的图片文件路径或PIL Image, 提取的图片文件路径列表或PIL Image列表)
    """
    matches, matches_image, matches_other = re_match(ocr_text)
    if not matches:
        logger.warning("未找到边界框信息")
        if save_to_disk:
            return None, []
        else:
            return image, []

    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    # 创建半透明覆盖层
    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)

    font = ImageFont.load_default()

    # 存储提取的图片
    extracted_images = []
    img_idx = 0

    for i, ref in enumerate(matches):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result

                # 随机颜色
                color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))
                color_a = color + (20,)  # 半透明

                for points in points_list:
                    x1, y1, x2, y2 = points

                    # 坐标归一化（DeepSeek-OCR 使用 0-999 范围）
                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)
                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)

                    # 提取图片区域（如果是 image 类型且需要提取）
                    if label_type == 'image' and extract_images:
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            extracted_images.append(cropped)
                            img_idx += 1
                        except Exception as e:
                            logger.error(f"提取图片失败: {e}")

                    try:
                        # 绘制边界框
                        if label_type == 'title':
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)

                        # 绘制标签
                        text_x = x1
                        text_y = max(0, y1 - 15)

                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height],
                                     fill=(255, 255, 255, 30))

                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except Exception as e:
                        logger.error(f"绘制边界框失败: {e}")
                        pass
        except Exception as e:
            logger.error(f"处理引用失败: {e}")
            continue

    # 合并覆盖层
    img_draw.paste(overlay, (0, 0), overlay)

    # 如果需要保存到磁盘
    if save_to_disk:
        import uuid
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]

        # 保存带边界框的图片
        boxes_filename = f"{filename_prefix}_boxes_{timestamp}_{unique_id}.jpg"
        boxes_path = OUTPUT_DIR / boxes_filename
        img_draw.save(boxes_path, 'JPEG', quality=95)
        logger.info(f"✅ 保存带边界框的图片: {boxes_path}")

        # 保存提取的图片
        extracted_paths = []
        for idx, extracted_img in enumerate(extracted_images):
            extracted_filename = f"{filename_prefix}_extracted_{idx+1}_{timestamp}_{unique_id}.jpg"
            extracted_path = OUTPUT_DIR / extracted_filename
            extracted_img.save(extracted_path, 'JPEG', quality=95)
            extracted_paths.append(f"/outputs/{extracted_filename}")
            logger.info(f"✅ 保存提取的图片: {extracted_path}")

        return f"/outputs/{boxes_filename}", extracted_paths
    else:
        return img_draw, extracted_images


def load_model():
    """加载 DeepSeek-OCR 模型"""
    global model, tokenizer, MODEL_LOADED, last_request_time

    with model_load_lock:
        if MODEL_LOADED:
            logger.info("模型已加载，跳过重复加载")
            return

        try:
            load_start = datetime.now()
            logger.info(f"🚀 开始加载模型: {MODEL_NAME}")
            logger.info(f"📍 使用设备: {DEVICE}")

            from transformers import AutoModel, AutoTokenizer

            logger.info("⏳ 加载 tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True
            )

            logger.info("⏳ 加载模型权重...")
            model = AutoModel.from_pretrained(
                MODEL_NAME,
                attn_implementation='eager',  # 使用 eager 模式，不使用 Flash Attention
                trust_remote_code=True,
                use_safetensors=True
            )

            logger.info("⏳ 移动模型到 GPU...")
            model = model.eval()
            if DEVICE == "cuda":
                # 先转换为 bfloat16，再移到 GPU（避免 CUDA 兼容性问题）
                model = model.to(torch.bfloat16).to(DEVICE)
            else:
                model = model.to(DEVICE)

            MODEL_LOADED = True
            last_request_time = datetime.now()

            load_time = (datetime.now() - load_start).total_seconds()
            logger.info(f"✅ 模型加载成功！耗时: {load_time:.2f} 秒")

            # 显示 GPU 内存使用情况
            if DEVICE == "cuda":
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"💾 GPU 内存: 已分配 {memory_allocated:.2f} GB, 已保留 {memory_reserved:.2f} GB")

        except Exception as e:
            logger.error(f"❌ 模型加载失败: {str(e)}")
            raise


def unload_model():
    """卸载模型释放 GPU 内存"""
    global model, tokenizer, MODEL_LOADED

    with model_load_lock:
        if not MODEL_LOADED:
            logger.info("模型未加载，无需卸载")
            return

        try:
            logger.info("🔄 开始卸载模型，释放 GPU 内存...")

            # 移动模型到 CPU 并删除
            if model is not None:
                if DEVICE == "cuda":
                    model = model.cpu()
                del model

            if tokenizer is not None:
                del tokenizer

            # 清理 GPU 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # 强制 Python 垃圾回收
            gc.collect()

            model = None
            tokenizer = None
            MODEL_LOADED = False

            # 显示释放后的 GPU 内存
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"💾 GPU 内存释放后: 已分配 {memory_allocated:.2f} GB, 已保留 {memory_reserved:.2f} GB")

            logger.info("✅ 模型已卸载，GPU 内存已释放")

        except Exception as e:
            logger.error(f"❌ 模型卸载失败: {str(e)}")


async def idle_monitor():
    """后台任务：监控空闲时间并自动卸载模型"""
    global last_request_time

    logger.info(f"🔍 空闲监控已启动，超时时间: {IDLE_TIMEOUT} 秒 ({IDLE_TIMEOUT/60:.1f} 分钟)")

    while True:
        await asyncio.sleep(60)  # 每分钟检查一次

        if MODEL_LOADED and last_request_time:
            idle_time = (datetime.now() - last_request_time).total_seconds()

            if idle_time > IDLE_TIMEOUT:
                logger.info(f"⏰ 模型空闲 {idle_time:.0f} 秒 ({idle_time/60:.1f} 分钟)，开始卸载...")
                unload_model()
                last_request_time = None


async def cleanup_old_files():
    """定期清理过期的输出文件"""
    while True:
        try:
            await asyncio.sleep(3600)  # 每小时检查一次

            current_time = datetime.now().timestamp()
            deleted_count = 0

            for file_path in OUTPUT_DIR.glob("*.jpg"):
                # 检查文件修改时间
                file_mtime = file_path.stat().st_mtime
                if current_time - file_mtime > OUTPUT_FILE_RETENTION:
                    file_path.unlink()
                    deleted_count += 1
                    logger.info(f"🗑️  删除过期文件: {file_path.name}")

            if deleted_count > 0:
                logger.info(f"✅ 清理完成，删除了 {deleted_count} 个过期文件")

        except Exception as e:
            logger.error(f"❌ 清理文件失败: {e}")


@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("🚀 正在启动 DeepSeek-OCR API 服务...")
    logger.info(f"⚙️  配置: LAZY_LOAD={LAZY_LOAD}, IDLE_TIMEOUT={IDLE_TIMEOUT}s ({IDLE_TIMEOUT/60:.1f}分钟)")
    logger.info(f"📁 输出目录: {OUTPUT_DIR.absolute()}")
    logger.info(f"🗑️  文件保留时间: {OUTPUT_FILE_RETENTION}s ({OUTPUT_FILE_RETENTION/3600:.1f}小时)")
    logger.info(f"🌐 API 基础 URL: {API_BASE_URL}")

    if LAZY_LOAD:
        logger.info("💤 按需加载模式：模型将在首次请求时加载")
        # 启动空闲监控任务
        asyncio.create_task(idle_monitor())
    else:
        logger.info("🔥 预加载模式：立即加载模型")
        load_model()

    # 启动文件清理任务
    asyncio.create_task(cleanup_old_files())

    logger.info("✅ 服务启动完成！")


@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "DeepSeek-OCR API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": MODEL_LOADED,
        "device": DEVICE
    }


@app.get("/health")
async def health_check():
    """健康检查（不触发模型加载）"""
    health_info = {
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "device": DEVICE,
        "gpu_available": torch.cuda.is_available(),
        "lazy_load": LAZY_LOAD,
        "idle_timeout": IDLE_TIMEOUT,
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
        "active_requests": active_requests,
        "max_file_size_mb": MAX_FILE_SIZE / 1024 / 1024,
        "timestamp": datetime.now().isoformat()
    }

    # 如果模型已加载，显示空闲时间
    if MODEL_LOADED and last_request_time:
        idle_time = (datetime.now() - last_request_time).total_seconds()
        health_info["idle_time_seconds"] = idle_time
        health_info["idle_time_minutes"] = idle_time / 60

    # 显示 GPU 内存使用
    if torch.cuda.is_available():
        health_info["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
        health_info["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3

    return health_info


@app.get("/outputs/{filename}")
async def download_output(filename: str):
    """
    下载生成的输出文件

    支持远程访问，前端可以通过此端点获取生成的图片

    Args:
        filename: 文件名（不包含路径）

    Returns:
        FileResponse: 图片文件

    Raises:
        HTTPException: 404 - 文件不存在或路径非法
    """
    # 安全检查：防止路径遍历攻击
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="非法的文件名")

    file_path = OUTPUT_DIR / filename

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")

    # 确保文件在 OUTPUT_DIR 目录内
    if not str(file_path.resolve()).startswith(str(OUTPUT_DIR.resolve())):
        raise HTTPException(status_code=403, detail="禁止访问")

    return FileResponse(
        path=file_path,
        media_type="image/jpeg",
        filename=filename,
        headers={
            "Cache-Control": "public, max-age=3600",  # 缓存 1 小时
            "Access-Control-Allow-Origin": "*"  # 允许跨域访问
        }
    )


@app.post("/ocr/image", response_model=OCRResponse)
async def ocr_image(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form("<image>\n<|grounding|>Convert the document to markdown."),
    base_size: Optional[int] = Form(1024),
    image_size: Optional[int] = Form(640),
    crop_mode: Optional[bool] = Form(True),
    save_results: Optional[bool] = Form(False),
    test_compress: Optional[bool] = Form(False),
    draw_boxes: Optional[bool] = Form(False),  # 已废弃，请使用 /ocr/image/boxes
    extract_images: Optional[bool] = Form(False)  # 已废弃，请使用 /ocr/image/extract
):
    """
    对上传的图片或 PDF 进行 OCR 识别（纯文本版）

    参数:
    - file: 图片文件或 PDF 文件
      - 图片支持: jpg, png, jpeg, bmp, gif, tiff, webp 等格式
      - PDF 支持: 自动识别并处理所有页面
    - prompt: OCR 提示词
    - base_size: 基础尺寸
    - image_size: 图片尺寸
    - crop_mode: 是否裁剪模式
    - save_results: 是否保存结果
    - test_compress: 是否测试压缩

    注意:
    - draw_boxes 和 extract_images 参数已废弃，将被忽略
    - 如需画框功能，请使用 /ocr/image/boxes
    - 如需提取图片功能，请使用 /ocr/image/extract
    - PDF 文件会自动处理所有页面，返回合并的文本结果
    """
    global last_request_time, active_requests

    # 使用并发控制信号量
    async with request_semaphore:
        # 更新活跃请求计数
        with active_requests_lock:
            active_requests += 1
            current_active = active_requests

        logger.info(f"📥 收到 OCR 请求，当前活跃请求数: {current_active}/{MAX_CONCURRENT_REQUESTS}")

        try:
            # 按需加载模型
            if not MODEL_LOADED:
                logger.info("🔥 检测到请求，开始加载模型...")
                load_model()

            start_time = datetime.now()
            temp_file = None

            try:
                # 读取文件内容
                content = await file.read()

                # 检测文件类型（支持图片和 PDF）
                is_pdf_file = is_pdf(content)

                if not is_pdf_file:
                    # 验证图片文件类型
                    if file.content_type and not file.content_type.startswith('image/'):
                        raise HTTPException(status_code=400, detail="只支持图片或 PDF 文件")

                    # 如果没有 content_type，通过文件扩展名验证
                    if not file.content_type:
                        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
                        filename = file.filename or "image.jpg"
                        file_ext = Path(filename).suffix.lower()
                        if file_ext not in allowed_extensions:
                            raise HTTPException(status_code=400, detail=f"不支持的文件类型: {file_ext}")

                # 检查文件大小
                file_size = len(content)
                if file_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=400,
                        detail=f"文件太大: {file_size / 1024 / 1024:.2f}MB，最大允许: {MAX_FILE_SIZE / 1024 / 1024:.2f}MB"
                    )

                filename = file.filename or "image.jpg"

                # 处理 PDF 或图片
                if is_pdf_file:
                    logger.info(f"📄 检测到 PDF 文件: {filename}, 大小: {file_size / 1024:.2f}KB")

                    # 将 PDF 转换为图片列表
                    images = pdf_to_images(content)
                    total_pages = len(images)
                    logger.info(f"📄 PDF 共 {total_pages} 页")

                    # 对每一页进行 OCR
                    all_results = []
                    for page_num, img in enumerate(images, 1):
                        logger.info(f"📝 处理第 {page_num}/{total_pages} 页...")

                        # 保存临时图片文件
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                            img.save(tmp.name, format='PNG')
                            temp_file = tmp.name

                        # 执行 OCR
                        if save_results:
                            output_path = str(OUTPUT_DIR)
                        else:
                            import tempfile as tmp_module
                            temp_output_dir = tmp_module.mkdtemp(prefix="deepseek_ocr_")
                            output_path = temp_output_dir

                        page_result = model.infer(
                            tokenizer,
                            prompt=prompt,
                            image_file=temp_file,
                            output_path=output_path,
                            base_size=base_size,
                            image_size=image_size,
                            crop_mode=crop_mode,
                            save_results=save_results,
                            test_compress=test_compress,
                            eval_mode=True
                        )

                        all_results.append(f"# 第 {page_num} 页\n\n{page_result}")

                        # 清理临时文件
                        try:
                            os.unlink(temp_file)
                        except:
                            pass

                    # 合并所有页的结果
                    result = "\n\n---\n\n".join(all_results)
                    temp_file = None  # 已经清理过了

                else:
                    # 处理图片文件
                    suffix = Path(filename).suffix or ".jpg"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(content)
                        temp_file = tmp.name

                    logger.info(f"🖼️  处理图片: {filename}, 大小: {file_size / 1024:.2f}KB")
                    logger.info(f"Prompt: {prompt}")
                    logger.info(f"eval_mode: True, save_results: {save_results}")

                    # 执行 OCR
                    if save_results:
                        output_path = str(OUTPUT_DIR)
                    else:
                        import tempfile as tmp_module
                        temp_output_dir = tmp_module.mkdtemp(prefix="deepseek_ocr_")
                        output_path = temp_output_dir

                    result = model.infer(
                        tokenizer,
                        prompt=prompt,
                        image_file=temp_file,
                        output_path=output_path,
                        base_size=base_size,
                        image_size=image_size,
                        crop_mode=crop_mode,
                        save_results=save_results,
                        test_compress=test_compress,
                        eval_mode=True
                    )

                # 更新最后请求时间
                last_request_time = datetime.now()

                processing_time = (datetime.now() - start_time).total_seconds()

                logger.info(f"✅ OCR 完成，耗时: {processing_time:.2f}s")

                # 清理 GPU 缓存
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()

                # 返回纯文本结果（忽略 draw_boxes 和 extract_images 参数）
                metadata = {
                    "filename": file.filename,
                    "file_size": file_size,
                    "base_size": base_size,
                    "image_size": image_size,
                    "crop_mode": crop_mode,
                }

                # 如果是 PDF，添加页数信息
                if is_pdf_file:
                    metadata["file_type"] = "pdf"
                    metadata["total_pages"] = total_pages
                else:
                    metadata["file_type"] = "image"

                return OCRResponse(
                    success=True,
                    text=result,
                    processing_time=processing_time,
                    metadata=metadata,
                    image_with_boxes_url=None,
                    extracted_images_urls=None
                )

            except torch.cuda.OutOfMemoryError as e:
                # GPU 内存溢出错误
                logger.error(f"❌ GPU 内存溢出: {str(e)}")
                logger.info("🧹 正在清理 GPU 缓存...")

                # 强制清理 GPU 缓存
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    gc.collect()

                raise HTTPException(
                    status_code=503,
                    detail=f"GPU 内存不足，请稍后重试。当前有 {current_active} 个并发请求正在处理。"
                )

            except Exception as e:
                logger.error(f"❌ OCR 处理失败: {str(e)}")
                raise HTTPException(status_code=500, detail=f"OCR 处理失败: {str(e)}")

            finally:
                # 清理临时文件
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except:
                        pass

        finally:
            # 减少活跃请求计数
            with active_requests_lock:
                active_requests -= 1
            logger.info(f"📤 请求完成，当前活跃请求数: {active_requests}/{MAX_CONCURRENT_REQUESTS}")


@app.post("/ocr/image/boxes", response_model=OCRBoxesResponse)
async def ocr_image_boxes(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form("<image>\n<|grounding|>Convert the document to markdown."),
    base_size: Optional[int] = Form(1024),
    image_size: Optional[int] = Form(640),
    crop_mode: Optional[bool] = Form(True),
    save_results: Optional[bool] = Form(False),
    test_compress: Optional[bool] = Form(False),
    include_text: Optional[bool] = Form(True)  # 是否返回完整文本
):
    """
    对上传的图片或 PDF 进行 OCR 识别并画框

    参数:
    - file: 图片文件或 PDF 文件
    - prompt: OCR 提示词
    - base_size: 基础尺寸
    - image_size: 图片尺寸
    - crop_mode: 是否裁剪模式
    - save_results: 是否保存结果
    - test_compress: 是否测试压缩
    - include_text: 是否返回完整 OCR 文本

    返回:
    - 画好框的图片 URL（PDF 时返回第一页的画框图片）
    - 所有边界框的结构化信息（PDF 时包含所有页）
    - 可选的完整 OCR 文本（PDF 时包含所有页）

    注意:
    - PDF 文件会处理所有页面，为每一页都画框
    - 返回的 image_with_boxes_url 是第一页的画框图片
    - metadata 中会包含 all_pages_urls 字段，包含所有页的画框图片 URL
    """
    global last_request_time, active_requests

    async with request_semaphore:
        with active_requests_lock:
            active_requests += 1
            current_active = active_requests

        logger.info(f"📥 收到画框请求，当前活跃请求数: {current_active}/{MAX_CONCURRENT_REQUESTS}")

        try:
            if not MODEL_LOADED:
                logger.info("🔥 检测到请求，开始加载模型...")
                load_model()

            start_time = datetime.now()
            temp_file = None

            try:
                # 读取文件内容
                content = await file.read()
                file_size = len(content)

                # 检测文件类型（支持图片和 PDF）
                is_pdf_file = is_pdf(content)

                if not is_pdf_file:
                    # 验证图片文件类型
                    if file.content_type and not file.content_type.startswith('image/'):
                        raise HTTPException(status_code=400, detail="只支持图片或 PDF 文件")

                    if not file.content_type:
                        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
                        filename = file.filename or "image.jpg"
                        file_ext = Path(filename).suffix.lower()
                        if file_ext not in allowed_extensions:
                            raise HTTPException(status_code=400, detail=f"不支持的文件类型: {file_ext}")

                if file_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=400,
                        detail=f"文件太大: {file_size / 1024 / 1024:.2f}MB，最大允许: {MAX_FILE_SIZE / 1024 / 1024:.2f}MB"
                    )

                filename = file.filename or "image.jpg"

                # 处理 PDF 或图片
                all_boxes = []
                all_text_parts = []
                all_pages_urls = []
                total_pages = 1

                if is_pdf_file:
                    logger.info(f"📄 检测到 PDF 文件: {filename}, 大小: {file_size / 1024:.2f}KB")

                    # 将 PDF 转换为图片列表
                    images = pdf_to_images(content)
                    total_pages = len(images)
                    logger.info(f"📄 PDF 共 {total_pages} 页")

                    # 对每一页进行 OCR 和画框
                    for page_num, img in enumerate(images, 1):
                        logger.info(f"📝 处理第 {page_num}/{total_pages} 页...")

                        # 保存临时图片文件
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                            img.save(tmp.name, format='PNG')
                            temp_file = tmp.name

                        # 执行 OCR
                        if save_results:
                            output_path = str(OUTPUT_DIR)
                        else:
                            import tempfile as tmp_module
                            temp_output_dir = tmp_module.mkdtemp(prefix="deepseek_ocr_")
                            output_path = temp_output_dir

                        page_result = model.infer(
                            tokenizer,
                            prompt=prompt,
                            image_file=temp_file,
                            output_path=output_path,
                            base_size=base_size,
                            image_size=image_size,
                            crop_mode=crop_mode,
                            save_results=save_results,
                            test_compress=test_compress,
                            eval_mode=True
                        )

                        all_text_parts.append(f"# 第 {page_num} 页\n\n{page_result}")

                        # 解析边界框
                        image_width, image_height = img.size
                        page_boxes = _parse_boxes_from_text(page_result, image_width, image_height)
                        all_boxes.extend(page_boxes)

                        # 画框
                        import uuid
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        unique_id = str(uuid.uuid4())[:8]
                        filename_prefix = f"{Path(filename).stem}_page{page_num}_{timestamp}_{unique_id}"

                        page_boxes_url, _ = draw_bounding_boxes(
                            img,
                            page_result,
                            extract_images=False,
                            save_to_disk=True,
                            filename_prefix=filename_prefix
                        )

                        if page_boxes_url:
                            page_boxes_url = f"{API_BASE_URL}{page_boxes_url}"
                            all_pages_urls.append(page_boxes_url)
                            logger.info(f"✅ 第 {page_num} 页边界框绘制完成: {page_boxes_url}")

                        # 清理临时文件
                        try:
                            os.unlink(temp_file)
                        except:
                            pass

                    result = "\n\n---\n\n".join(all_text_parts)
                    boxes = all_boxes
                    image_with_boxes_url = all_pages_urls[0] if all_pages_urls else None
                    temp_file = None  # 已经清理过了

                else:
                    # 处理图片文件
                    suffix = Path(filename).suffix or ".jpg"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(content)
                        temp_file = tmp.name

                    logger.info(f"🖼️  处理图片: {filename}, 大小: {file_size / 1024:.2f}KB")

                    # 执行 OCR
                    if save_results:
                        output_path = str(OUTPUT_DIR)
                    else:
                        import tempfile as tmp_module
                        temp_output_dir = tmp_module.mkdtemp(prefix="deepseek_ocr_")
                        output_path = temp_output_dir

                    result = model.infer(
                        tokenizer,
                        prompt=prompt,
                        image_file=temp_file,
                        output_path=output_path,
                        base_size=base_size,
                        image_size=image_size,
                        crop_mode=crop_mode,
                        save_results=save_results,
                        test_compress=test_compress,
                        eval_mode=True
                    )

                    # 解析边界框
                    original_image = Image.open(temp_file).convert('RGB')
                    image_width, image_height = original_image.size
                    boxes = _parse_boxes_from_text(result, image_width, image_height)

                    # 画框
                    import uuid
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    unique_id = str(uuid.uuid4())[:8]
                    filename_prefix = f"{Path(filename).stem}_{timestamp}_{unique_id}"

                    image_with_boxes_url, _ = draw_bounding_boxes(
                        original_image,
                        result,
                        extract_images=False,
                        save_to_disk=True,
                        filename_prefix=filename_prefix
                    )

                # 更新最后请求时间和计算处理时间
                last_request_time = datetime.now()
                processing_time = (datetime.now() - start_time).total_seconds()

                logger.info(f"✅ 画框完成，耗时: {processing_time:.2f}s")

                # 转换为完整 URL（图片文件）
                if not is_pdf_file and image_with_boxes_url:
                    image_with_boxes_url = f"{API_BASE_URL}{image_with_boxes_url}"
                    logger.info(f"✅ 边界框绘制完成: {image_with_boxes_url}")

                # 生成 labels_summary
                labels_summary = sorted(list(set(box.label_type for box in boxes)))

                # 清理 GPU 缓存
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()

                # 构建 metadata
                metadata = {
                    "filename": filename,
                    "file_size": file_size,
                    "base_size": base_size,
                    "image_size": image_size,
                    "crop_mode": crop_mode,
                    "boxes_count": len(boxes)
                }

                # 如果是 PDF，添加页数信息和所有页的 URL
                if is_pdf_file:
                    metadata["file_type"] = "pdf"
                    metadata["total_pages"] = total_pages
                    metadata["all_pages_urls"] = all_pages_urls
                else:
                    metadata["file_type"] = "image"

                return OCRBoxesResponse(
                    success=True,
                    image_with_boxes_url=image_with_boxes_url or "",
                    boxes=boxes,
                    text=result if include_text else None,
                    question=prompt,
                    labels_summary=labels_summary,
                    processing_time=processing_time,
                    metadata=metadata
                )

            except Exception as e:
                logger.error(f"❌ 画框处理失败: {str(e)}")
                raise HTTPException(status_code=500, detail=f"画框处理失败: {str(e)}")

            finally:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except:
                        pass

        finally:
            with active_requests_lock:
                active_requests -= 1
            logger.info(f"📤 请求完成，当前活跃请求数: {active_requests}/{MAX_CONCURRENT_REQUESTS}")


@app.post("/ocr/image/extract", response_model=OCRExtractResponse)
async def ocr_image_extract(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form("<image>\n<|grounding|>Convert the document to markdown."),
    base_size: Optional[int] = Form(1024),
    image_size: Optional[int] = Form(640),
    crop_mode: Optional[bool] = Form(True),
    save_results: Optional[bool] = Form(False),
    test_compress: Optional[bool] = Form(False),
    include_text: Optional[bool] = Form(True),  # 是否返回完整文本
    include_boxes: Optional[bool] = Form(False)  # 是否返回边界框信息
):
    """
    对上传的图片或 PDF 进行 OCR 识别并提取图片区域

    参数:
    - file: 图片文件或 PDF 文件
    - prompt: OCR 提示词
    - base_size: 基础尺寸
    - image_size: 图片尺寸
    - crop_mode: 是否裁剪模式
    - save_results: 是否保存结果
    - test_compress: 是否测试压缩
    - include_text: 是否返回完整 OCR 文本
    - include_boxes: 是否返回边界框信息

    返回:
    - 提取的图片区域列表（每个区域包含子图 URL、边界框、对应文字）
    - 可选的完整 OCR 文本

    注意:
    - PDF 文件会处理所有页面，提取所有页面中的图片区域
    """
    global last_request_time, active_requests

    async with request_semaphore:
        with active_requests_lock:
            active_requests += 1
            current_active = active_requests

        logger.info(f"📥 收到提取请求，当前活跃请求数: {current_active}/{MAX_CONCURRENT_REQUESTS}")

        try:
            if not MODEL_LOADED:
                logger.info("🔥 检测到请求，开始加载模型...")
                load_model()

            start_time = datetime.now()
            temp_file = None

            try:
                # 读取文件内容
                content = await file.read()
                file_size = len(content)

                # 检测文件类型（支持图片和 PDF）
                is_pdf_file = is_pdf(content)

                if not is_pdf_file:
                    # 验证图片文件类型
                    if file.content_type and not file.content_type.startswith('image/'):
                        raise HTTPException(status_code=400, detail="只支持图片或 PDF 文件")

                    if not file.content_type:
                        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
                        filename = file.filename or "image.jpg"
                        file_ext = Path(filename).suffix.lower()
                        if file_ext not in allowed_extensions:
                            raise HTTPException(status_code=400, detail=f"不支持的文件类型: {file_ext}")

                if file_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=400,
                        detail=f"文件太大: {file_size / 1024 / 1024:.2f}MB，最大允许: {MAX_FILE_SIZE / 1024 / 1024:.2f}MB"
                    )

                filename = file.filename or "image.jpg"

                # 处理 PDF 或图片
                all_regions = []
                all_text_parts = []
                total_pages = 1

                if is_pdf_file:
                    logger.info(f"📄 检测到 PDF 文件: {filename}, 大小: {file_size / 1024:.2f}KB")

                    # 将 PDF 转换为图片列表
                    images = pdf_to_images(content)
                    total_pages = len(images)
                    logger.info(f"📄 PDF 共 {total_pages} 页")

                    # 对每一页进行 OCR 和提取
                    for page_num, img in enumerate(images, 1):
                        logger.info(f"📝 处理第 {page_num}/{total_pages} 页...")

                        # 保存临时图片文件
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                            img.save(tmp.name, format='PNG')
                            temp_file = tmp.name

                        # 执行 OCR
                        if save_results:
                            output_path = str(OUTPUT_DIR)
                        else:
                            import tempfile as tmp_module
                            temp_output_dir = tmp_module.mkdtemp(prefix="deepseek_ocr_")
                            output_path = temp_output_dir

                        page_result = model.infer(
                            tokenizer,
                            prompt=prompt,
                            image_file=temp_file,
                            output_path=output_path,
                            base_size=base_size,
                            image_size=image_size,
                            crop_mode=crop_mode,
                            save_results=save_results,
                            test_compress=test_compress,
                            eval_mode=True
                        )

                        all_text_parts.append(f"# 第 {page_num} 页\n\n{page_result}")

                        # 解析边界框并提取图片区域
                        image_width, image_height = img.size
                        boxes = _parse_boxes_from_text(page_result, image_width, image_height)

                        # 提取图片区域
                        import uuid
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        unique_id = str(uuid.uuid4())[:8]
                        filename_prefix = f"{Path(filename).stem}_page{page_num}_{timestamp}_{unique_id}"

                        for box in boxes:
                            if box.label_type == "image":
                                try:
                                    cropped = img.crop((box.x1_px, box.y1_px, box.x2_px, box.y2_px))
                                    extracted_filename = f"{filename_prefix}_extracted_{len(all_regions)+1}.jpg"
                                    extracted_path = OUTPUT_DIR / extracted_filename
                                    cropped.save(extracted_path, 'JPEG', quality=95)

                                    image_url = f"{API_BASE_URL}/outputs/{extracted_filename}"
                                    region_text = _extract_region_text(page_result, len([r for r in all_regions if r.label_type == "image"]))

                                    region = Region(
                                        id=f"{box.id}_page{page_num}",
                                        label_type=box.label_type,
                                        page_number=page_num,
                                        bbox=box if include_boxes else None,
                                        image_url=image_url,
                                        text=region_text
                                    )
                                    all_regions.append(region)
                                    logger.info(f"✅ 提取图片区域: {extracted_filename}")
                                except Exception as e:
                                    logger.error(f"❌ 提取图片区域失败: {e}")
                                    continue

                        # 清理临时文件
                        try:
                            os.unlink(temp_file)
                        except:
                            pass

                    result = "\n\n---\n\n".join(all_text_parts)
                    regions = all_regions
                    temp_file = None  # 已经清理过了

                else:
                    # 处理图片文件
                    suffix = Path(filename).suffix or ".jpg"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(content)
                        temp_file = tmp.name

                    logger.info(f"🖼️  处理图片: {filename}, 大小: {file_size / 1024:.2f}KB")

                    # 执行 OCR
                    if save_results:
                        output_path = str(OUTPUT_DIR)
                    else:
                        import tempfile as tmp_module
                        temp_output_dir = tmp_module.mkdtemp(prefix="deepseek_ocr_")
                        output_path = temp_output_dir

                    result = model.infer(
                        tokenizer,
                        prompt=prompt,
                        image_file=temp_file,
                        output_path=output_path,
                        base_size=base_size,
                        image_size=image_size,
                        crop_mode=crop_mode,
                        save_results=save_results,
                        test_compress=test_compress,
                        eval_mode=True
                    )

                    # 解析边界框
                    original_image = Image.open(temp_file).convert('RGB')
                    image_width, image_height = original_image.size
                    boxes = _parse_boxes_from_text(result, image_width, image_height)

                    # 提取图片区域
                    import uuid
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    unique_id = str(uuid.uuid4())[:8]
                    filename_prefix = f"{Path(filename).stem}_{timestamp}_{unique_id}"

                    regions = []
                    region_idx = 0

                    for box in boxes:
                        # 只提取 label_type 为 "image" 的区域
                        if box.label_type == "image":
                            try:
                                # 裁剪图片
                                cropped = original_image.crop((box.x1_px, box.y1_px, box.x2_px, box.y2_px))

                                # 保存裁剪的图片
                                extracted_filename = f"{filename_prefix}_extracted_{region_idx+1}_{timestamp}_{unique_id}.jpg"
                                extracted_path = OUTPUT_DIR / extracted_filename
                                cropped.save(extracted_path, 'JPEG', quality=95)

                                image_url = f"{API_BASE_URL}/outputs/{extracted_filename}"

                                # 提取该区域的文字
                                region_text = _extract_region_text(result, region_idx)

                                region = Region(
                                    id=box.id,
                                    label_type=box.label_type,
                                    bbox=box if include_boxes else None,
                                    image_url=image_url,
                                    text=region_text
                                )
                                regions.append(region)
                                region_idx += 1

                                logger.info(f"✅ 提取图片区域: {extracted_filename}")
                            except Exception as e:
                                logger.error(f"❌ 提取图片区域失败: {e}")
                                continue

                    logger.info(f"✅ 共提取了 {len(regions)} 个图片区域")

                # 更新最后请求时间和计算处理时间
                last_request_time = datetime.now()
                processing_time = (datetime.now() - start_time).total_seconds()

                logger.info(f"✅ 提取完成，耗时: {processing_time:.2f}s，共提取 {len(regions)} 个区域")

                # 清理 GPU 缓存
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()

                # 构建 metadata
                metadata = {
                    "filename": filename,
                    "file_size": file_size,
                    "base_size": base_size,
                    "image_size": image_size,
                    "crop_mode": crop_mode,
                    "regions_count": len(regions)
                }

                # 如果是 PDF，添加页数信息
                if is_pdf_file:
                    metadata["file_type"] = "pdf"
                    metadata["total_pages"] = total_pages
                else:
                    metadata["file_type"] = "image"

                return OCRExtractResponse(
                    success=True,
                    text=result if include_text else None,
                    regions=regions,
                    question=prompt,
                    processing_time=processing_time,
                    metadata=metadata
                )

            except Exception as e:
                logger.error(f"❌ 提取处理失败: {str(e)}")
                raise HTTPException(status_code=500, detail=f"提取处理失败: {str(e)}")

            finally:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except:
                        pass

        finally:
            with active_requests_lock:
                active_requests -= 1
            logger.info(f"📤 请求完成，当前活跃请求数: {active_requests}/{MAX_CONCURRENT_REQUESTS}")


@app.post("/ocr/batch", response_model=List[OCRResponse])
async def ocr_batch(
    files: List[UploadFile] = File(...),
    prompt: Optional[str] = Form("<image>\n<|grounding|>Convert the document to markdown."),
    base_size: Optional[int] = Form(1024),
    image_size: Optional[int] = Form(640),
    crop_mode: Optional[bool] = Form(True)
):
    """
    批量处理多个图片
    """
    global last_request_time

    # 按需加载模型
    if not MODEL_LOADED:
        logger.info("🔥 检测到批量请求，开始加载模型...")
        load_model()
    
    results = []
    
    for file in files:
        result = await ocr_image(
            file=file,
            prompt=prompt,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            save_results=False,
            test_compress=True
        )
        results.append(result)
    
    return results


@app.post("/ocr/base64", response_model=OCRResponse)
async def ocr_base64(
    image_base64: str = Form(...),
    prompt: Optional[str] = Form("<image>\n<|grounding|>Convert the document to markdown."),
    base_size: Optional[int] = Form(1024),
    image_size: Optional[int] = Form(640),
    crop_mode: Optional[bool] = Form(True)
):
    """
    对 Base64 编码的图片进行 OCR 识别
    """
    global last_request_time

    # 按需加载模型
    if not MODEL_LOADED:
        logger.info("🔥 检测到 Base64 请求，开始加载模型...")
        load_model()
    
    start_time = datetime.now()
    temp_file = None
    
    try:
        # 解码 Base64
        image_data = base64.b64decode(image_base64)
        
        # 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            tmp.write(image_data)
            temp_file = tmp.name
        
        # 执行 OCR
        result = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=temp_file,
            output_path=None,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            save_results=False,
            test_compress=True
        )

        # 更新最后请求时间
        last_request_time = datetime.now()

        processing_time = (datetime.now() - start_time).total_seconds()
        
        return OCRResponse(
            success=True,
            text=result,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"OCR 处理失败: {str(e)}")
        return OCRResponse(
            success=False,
            error=str(e)
        )
    
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except:
                pass


@app.post("/admin/unload")
async def admin_unload():
    """手动卸载模型（管理端点）"""
    if not MODEL_LOADED:
        return {"status": "info", "message": "模型未加载，无需卸载"}

    unload_model()
    return {"status": "success", "message": "模型已卸载，GPU 内存已释放"}


@app.post("/admin/load")
async def admin_load():
    """手动加载模型（管理端点）"""
    if MODEL_LOADED:
        return {"status": "info", "message": "模型已加载"}

    load_model()
    return {"status": "success", "message": "模型已加载"}


@app.get("/admin/status")
async def admin_status():
    """获取详细状态信息（管理端点）"""
    status = {
        "model_loaded": MODEL_LOADED,
        "device": DEVICE,
        "lazy_load": LAZY_LOAD,
        "idle_timeout": IDLE_TIMEOUT,
        "idle_timeout_minutes": IDLE_TIMEOUT / 60,
    }

    if last_request_time:
        idle_time = (datetime.now() - last_request_time).total_seconds()
        status["last_request_time"] = last_request_time.isoformat()
        status["idle_time_seconds"] = idle_time
        status["idle_time_minutes"] = idle_time / 60
        status["will_unload_in_seconds"] = max(0, IDLE_TIMEOUT - idle_time)

    if torch.cuda.is_available():
        status["gpu_available"] = True
        status["gpu_name"] = torch.cuda.get_device_name(0)
        status["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
        status["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
        status["gpu_memory_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
    else:
        status["gpu_available"] = False

    return status


if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8200,  # 修改为非主流端口避免冲突
        reload=False,
        workers=1,
        timeout_keep_alive=300,  # 保持连接 5 分钟
        timeout_graceful_shutdown=30,  # 优雅关闭超时 30 秒
        limit_concurrency=10,  # 限制并发连接数
        backlog=2048  # 增加积压队列
    )

