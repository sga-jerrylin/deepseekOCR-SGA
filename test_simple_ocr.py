"""
简单的 OCR 测试脚本
"""
import requests
import time

API_URL = "http://localhost:8200"

def test_ocr():
    """测试 OCR 功能"""
    print("\n🚀 测试 OCR 功能\n")
    
    # 1. 健康检查
    print("1. 健康检查...")
    response = requests.get(f"{API_URL}/health")
    print(f"   状态: {response.status_code}")
    print(f"   响应: {response.json()}\n")
    
    # 2. OCR 测试
    print("2. OCR 识别测试...")
    print("   上传图片: test_image.jpg")
    
    with open("test_image.jpg", "rb") as f:
        files = {"file": ("test_image.jpg", f, "image/jpeg")}
        # 使用默认 prompt,包含 <image> 标记
        data = {}  # 不传 prompt,使用默认值
        
        print("   发送请求...")
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{API_URL}/ocr/image",
                files=files,
                data=data,
                timeout=180
            )
            
            elapsed = time.time() - start_time
            
            print(f"   状态码: {response.status_code}")
            print(f"   耗时: {elapsed:.2f} 秒")
            
            result = response.json()
            print(f"\n   完整响应:")
            import json
            print(json.dumps(result, indent=2, ensure_ascii=False))
                
        except requests.exceptions.Timeout:
            print("   ❌ 请求超时!")
        except Exception as e:
            print(f"   ❌ 错误: {e}")

if __name__ == "__main__":
    test_ocr()

