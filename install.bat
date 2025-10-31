@echo off
echo ========================================
echo DeepSeek-OCR 环境安装
echo ========================================

echo.
echo 步骤 1: 创建 Conda 环境
call conda create -n deepseek-ocr python=3.12.9 -y
if errorlevel 1 (
    echo 错误: Conda 环境创建失败
    pause
    exit /b 1
)

echo.
echo 步骤 2: 激活环境并安装 PyTorch
call conda activate deepseek-ocr
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

echo.
echo 步骤 3: 安装基础依赖
pip install -r requirements.txt

echo.
echo 步骤 4: 安装 flash-attention
pip install flash-attn==2.7.3 --no-build-isolation

echo.
echo 步骤 5: 安装 API 服务依赖
pip install fastapi==0.109.0 uvicorn[standard]==0.27.0 python-multipart==0.0.6 aiofiles==23.2.1 requests

echo.
echo 步骤 6: 创建输出目录
if not exist "outputs" mkdir outputs
if not exist "models" mkdir models

echo.
echo ========================================
echo 安装完成！
echo ========================================
echo.
echo 使用说明:
echo 1. 激活环境: conda activate deepseek-ocr
echo 2. 启动服务: python api_server.py
echo 3. 测试 API: python test_api.py
echo 4. 访问文档: http://localhost:8000/docs
echo.
pause

