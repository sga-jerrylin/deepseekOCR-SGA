@echo off
echo ========================================
echo 启动 DeepSeek-OCR API 服务
echo ========================================
echo.

call conda activate deepseek-ocr
if errorlevel 1 (
    echo 错误: 无法激活 conda 环境
    echo 请先运行 install.bat 安装环境
    pause
    exit /b 1
)

echo 正在启动服务...
echo 服务地址: http://localhost:8000
echo API 文档: http://localhost:8000/docs
echo.
echo 按 Ctrl+C 停止服务
echo.

python api_server.py

