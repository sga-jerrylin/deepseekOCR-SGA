# DeepSeek-OCR 环境安装脚本 (Windows PowerShell)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "DeepSeek-OCR 环境安装" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 检查 conda 是否安装
Write-Host "`n检查 Conda 环境..." -ForegroundColor Yellow
$condaPath = Get-Command conda -ErrorAction SilentlyContinue
if (-not $condaPath) {
    Write-Host "错误: 未找到 Conda，请先安装 Anaconda 或 Miniconda" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Conda 已安装" -ForegroundColor Green

# 检查 CUDA
Write-Host "`n检查 CUDA 环境..." -ForegroundColor Yellow
try {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        Write-Host "NVIDIA GPU 驱动已安装" -ForegroundColor Green
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    } else {
        Write-Host "警告: 未检测到 NVIDIA GPU 驱动" -ForegroundColor Yellow
    }
} catch {
    Write-Host "警告: 未检测到 NVIDIA GPU 驱动" -ForegroundColor Yellow
}

# 创建 conda 环境
Write-Host "`n创建 Conda 环境 (deepseek-ocr)..." -ForegroundColor Yellow
$envExists = conda env list | Select-String "deepseek-ocr"
if ($envExists) {
    Write-Host "环境已存在，是否删除并重新创建? (y/n): " -NoNewline -ForegroundColor Yellow
    $response = Read-Host
    if ($response -eq 'y') {
        conda env remove -n deepseek-ocr -y
        conda create -n deepseek-ocr python=3.12.9 -y
    }
} else {
    conda create -n deepseek-ocr python=3.12.9 -y
}
Write-Host "Conda 环境创建完成" -ForegroundColor Green

# 激活环境并安装依赖
Write-Host "`n安装 Python 依赖..." -ForegroundColor Yellow
Write-Host "这可能需要几分钟时间，请耐心等待..." -ForegroundColor Yellow

# 创建临时安装脚本
$installScript = @"
# 激活环境
conda activate deepseek-ocr

# 安装 PyTorch
Write-Host "安装 PyTorch..." -ForegroundColor Yellow
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# 安装基础依赖
Write-Host "安装基础依赖..." -ForegroundColor Yellow
pip install -r requirements.txt

# 安装 flash-attention
Write-Host "安装 flash-attention..." -ForegroundColor Yellow
pip install flash-attn==2.7.3 --no-build-isolation

# 安装 API 服务依赖
Write-Host "安装 API 服务依赖..." -ForegroundColor Yellow
pip install fastapi==0.109.0 uvicorn[standard]==0.27.0 python-multipart==0.0.6 aiofiles==23.2.1 requests

Write-Host "所有依赖安装完成" -ForegroundColor Green
"@

$installScript | Out-File -FilePath "temp_install.ps1" -Encoding UTF8
& conda run -n deepseek-ocr powershell -File "temp_install.ps1"
Remove-Item "temp_install.ps1"

# 创建输出目录
Write-Host "`n创建输出目录..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "outputs" | Out-Null
New-Item -ItemType Directory -Force -Path "models" | Out-Null
Write-Host "输出目录创建完成" -ForegroundColor Green

# 显示完成信息
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "环境安装完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`n使用说明:" -ForegroundColor Yellow
Write-Host "1. 激活环境: conda activate deepseek-ocr" -ForegroundColor White
Write-Host "2. 启动 API 服务: python api_server.py" -ForegroundColor White
Write-Host "3. 测试 API: python test_api.py" -ForegroundColor White
Write-Host "4. 访问 API 文档: http://localhost:8000/docs" -ForegroundColor White
Write-Host "`nDocker 部署:" -ForegroundColor Yellow
Write-Host "1. 构建镜像: docker-compose build" -ForegroundColor White
Write-Host "2. 启动服务: docker-compose up -d" -ForegroundColor White
Write-Host "3. 查看日志: docker-compose logs -f" -ForegroundColor White
Write-Host "4. 停止服务: docker-compose down" -ForegroundColor White

