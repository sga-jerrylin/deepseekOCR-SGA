"""
DeepSeek-OCR GPU 性能测试脚本
测试 API 响应时间和 GPU 使用情况
"""

import requests
import time
import base64
from pathlib import Path
import json

API_URL = "http://localhost:8200"

def test_health():
    """测试健康检查"""
    print("=" * 60)
    print("1. 健康检查测试")
    print("=" * 60)
    
    response = requests.get(f"{API_URL}/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()

def test_ocr_with_sample():
    """使用示例图片测试 OCR"""
    print("=" * 60)
    print("2. OCR 性能测试")
    print("=" * 60)
    
    # 创建一个简单的测试图片（如果没有的话）
    test_image_path = Path("test_image.jpg")
    
    if not test_image_path.exists():
        print("⚠️ 未找到测试图片 test_image.jpg")
        print("请提供一张测试图片，或者使用以下命令创建一个简单的测试图片：")
        print("  from PIL import Image, ImageDraw, ImageFont")
        print("  img = Image.new('RGB', (800, 600), color='white')")
        print("  img.save('test_image.jpg')")
        return
    
    # 测试单次 OCR
    print(f"\n📄 测试图片: {test_image_path}")
    print("⏱️  开始 OCR 识别...")
    
    start_time = time.time()
    
    with open(test_image_path, 'rb') as f:
        files = {'file': f}
        data = {'prompt': '<image>\n<|grounding|>Convert the document to markdown.'}
        response = requests.post(f"{API_URL}/ocr/image", files=files, data=data)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"✅ 识别完成！")
    print(f"⏱️  耗时: {elapsed_time:.2f} 秒")
    print(f"📊 状态码: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n📝 完整响应:")
        print("-" * 60)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("-" * 60)

        if 'text' in result and result['text']:
            print(f"\n📄 识别文本:")
            print(result['text'])

        if 'processing_time' in result and result['processing_time']:
            print(f"\n⚡ 服务器处理时间: {result['processing_time']:.2f} 秒")
    else:
        print(f"❌ 错误: {response.text}")
    
    print()

def test_multiple_requests():
    """测试多次请求的性能"""
    print("=" * 60)
    print("3. 连续请求性能测试")
    print("=" * 60)
    
    test_image_path = Path("test_image.jpg")
    
    if not test_image_path.exists():
        print("⚠️ 跳过此测试（需要 test_image.jpg）")
        return
    
    num_requests = 3
    print(f"\n🔄 将发送 {num_requests} 次连续请求...\n")
    
    times = []
    
    for i in range(num_requests):
        print(f"请求 {i+1}/{num_requests}...")
        start_time = time.time()
        
        with open(test_image_path, 'rb') as f:
            files = {'file': f}
            data = {'prompt': '<image>\n<|grounding|>Convert the document to markdown.'}
            response = requests.post(f"{API_URL}/ocr/image", files=files, data=data)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        times.append(elapsed_time)
        
        print(f"  ✅ 完成，耗时: {elapsed_time:.2f} 秒")
    
    print(f"\n📊 性能统计:")
    print(f"  平均耗时: {sum(times)/len(times):.2f} 秒")
    print(f"  最快: {min(times):.2f} 秒")
    print(f"  最慢: {max(times):.2f} 秒")
    print()

def print_gpu_monitoring_tip():
    """打印 GPU 监控提示"""
    print("=" * 60)
    print("4. GPU 监控提示")
    print("=" * 60)
    print("\n💡 在另一个终端运行以下命令来实时监控 GPU:")
    print("   nvidia-smi -l 1")
    print("\n或者使用:")
    print("   watch -n 1 nvidia-smi")
    print("\n观察以下指标:")
    print("  - GPU-Util: GPU 利用率（处理时应该接近 100%）")
    print("  - Memory-Usage: 显存使用（应该在 7-10 GB 之间）")
    print("  - Temp: 温度（正常范围 40-80°C）")
    print("  - Power: 功耗（处理时应该接近 180W）")
    print()

if __name__ == "__main__":
    print("\n🚀 DeepSeek-OCR GPU 性能测试\n")
    
    try:
        # 1. 健康检查
        test_health()
        
        # 2. 单次 OCR 测试
        test_ocr_with_sample()
        
        # 3. 多次请求测试
        test_multiple_requests()
        
        # 4. GPU 监控提示
        print_gpu_monitoring_tip()
        
        print("=" * 60)
        print("✅ 测试完成！")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到 API 服务")
        print("请确保 Docker 容器正在运行:")
        print("  docker ps")
        print("  docker-compose up -d")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

