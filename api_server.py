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
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import torch
import os
import tempfile
import base64
from pathlib import Path
import logging
from datetime import datetime
import uvicorn

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

# 配置
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-OCR")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs"))
OUTPUT_DIR.mkdir(exist_ok=True)


class OCRRequest(BaseModel):
    """OCR 请求模型"""
    prompt: Optional[str] = "<image>\n<|grounding|>Convert the document to markdown."
    base_size: Optional[int] = 1024
    image_size: Optional[int] = 640
    crop_mode: Optional[bool] = True
    save_results: Optional[bool] = False
    test_compress: Optional[bool] = True


class OCRResponse(BaseModel):
    """OCR 响应模型"""
    success: bool
    text: Optional[str] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
    metadata: Optional[dict] = None


def load_model():
    """加载 DeepSeek-OCR 模型"""
    global model, tokenizer, MODEL_LOADED
    
    if MODEL_LOADED:
        return
    
    try:
        logger.info(f"开始加载模型: {MODEL_NAME}")
        logger.info(f"使用设备: {DEVICE}")
        
        from transformers import AutoModel, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
        
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            attn_implementation='eager',  # 使用 eager 模式，不使用 Flash Attention
            trust_remote_code=True,
            use_safetensors=True
        )
        
        model = model.eval()
        if DEVICE == "cuda":
            # 先转换为 bfloat16，再移到 GPU（避免 CUDA 兼容性问题）
            model = model.to(torch.bfloat16).to(DEVICE)
        else:
            model = model.to(DEVICE)
        
        MODEL_LOADED = True
        logger.info("模型加载成功！")
        
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise


@app.on_event("startup")
async def startup_event():
    """应用启动时加载模型"""
    logger.info("正在启动 DeepSeek-OCR API 服务...")
    load_model()
    logger.info("服务启动完成！")


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
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "device": DEVICE,
        "gpu_available": torch.cuda.is_available(),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/ocr/image", response_model=OCRResponse)
async def ocr_image(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form("<image>\n<|grounding|>Convert the document to markdown."),
    base_size: Optional[int] = Form(1024),
    image_size: Optional[int] = Form(640),
    crop_mode: Optional[bool] = Form(True),
    save_results: Optional[bool] = Form(False),
    test_compress: Optional[bool] = Form(False)  # 改为 False,避免影响 eval_mode 的返回
):
    """
    对上传的图片进行 OCR 识别
    
    参数:
    - file: 图片文件 (支持 jpg, png, jpeg, bmp 等格式)
    - prompt: OCR 提示词
    - base_size: 基础尺寸
    - image_size: 图片尺寸
    - crop_mode: 是否裁剪模式
    - save_results: 是否保存结果
    - test_compress: 是否测试压缩
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    start_time = datetime.now()
    temp_file = None
    
    try:
        # 验证文件类型
        if file.content_type and not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="只支持图片文件")

        # 如果没有 content_type，通过文件扩展名验证
        if not file.content_type:
            allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
            filename = file.filename or "image.jpg"  # 提供默认文件名
            file_ext = Path(filename).suffix.lower()
            if file_ext not in allowed_extensions:
                raise HTTPException(status_code=400, detail=f"不支持的文件类型: {file_ext}")

        # 保存临时文件
        filename = file.filename or "image.jpg"  # 提供默认文件名
        suffix = Path(filename).suffix or ".jpg"  # 提供默认后缀
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_file = tmp.name

        logger.info(f"处理图片: {filename}, 大小: {len(content)} bytes")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"eval_mode: True, save_results: {save_results}")

        # 执行 OCR
        # 注意：output_path 不能为 None，infer 方法内部会调用 os.makedirs
        # 如果不保存结果，使用临时目录
        if save_results:
            output_path = str(OUTPUT_DIR)
        else:
            # 创建临时目录用于 infer 方法的内部处理
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
            eval_mode=True  # 启用 eval_mode 以返回识别文本
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        logger.info(f"OCR 完成，耗时: {processing_time:.2f}s")
        logger.info(f"识别结果类型: {type(result)}, 值: {result}")
        
        return OCRResponse(
            success=True,
            text=result,
            processing_time=processing_time,
            metadata={
                "filename": file.filename,
                "file_size": len(content),
                "base_size": base_size,
                "image_size": image_size,
                "crop_mode": crop_mode
            }
        )
        
    except Exception as e:
        logger.error(f"OCR 处理失败: {str(e)}")
        return OCRResponse(
            success=False,
            error=str(e)
        )
    
    finally:
        # 清理临时文件
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except:
                pass


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
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="模型未加载")
    
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
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="模型未加载")
    
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


if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8200,  # 修改为非主流端口避免冲突
        reload=False,
        workers=1
    )

