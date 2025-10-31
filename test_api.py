"""
DeepSeek-OCR API 测试脚本
"""

import requests
import base64
import json
from pathlib import Path
import time

# API 基础 URL
BASE_URL = "http://localhost:8000"


def test_health():
    """测试健康检查接口"""
    print("\n=== 测试健康检查 ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    return response.status_code == 200


def test_root():
    """测试根路径"""
    print("\n=== 测试根路径 ===")
    response = requests.get(f"{BASE_URL}/")
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    return response.status_code == 200


def test_ocr_image(image_path: str):
    """测试图片 OCR"""
    print(f"\n=== 测试图片 OCR: {image_path} ===")
    
    if not Path(image_path).exists():
        print(f"错误: 图片文件不存在: {image_path}")
        return False
    
    with open(image_path, 'rb') as f:
        files = {'file': (Path(image_path).name, f, 'image/jpeg')}
        data = {
            'prompt': '<image>\n<|grounding|>Convert the document to markdown.',
            'base_size': 1024,
            'image_size': 640,
            'crop_mode': True,
            'save_results': False,
            'test_compress': True
        }
        
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/ocr/image", files=files, data=data)
        elapsed_time = time.time() - start_time
        
        print(f"状态码: {response.status_code}")
        print(f"请求耗时: {elapsed_time:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            print(f"成功: {result['success']}")
            if result['success']:
                print(f"处理时间: {result['processing_time']:.2f}s")
                print(f"识别文本长度: {len(result['text'])} 字符")
                print(f"识别文本预览:\n{result['text'][:500]}...")
            else:
                print(f"错误: {result['error']}")
            return result['success']
        else:
            print(f"请求失败: {response.text}")
            return False


def test_ocr_base64(image_path: str):
    """测试 Base64 图片 OCR"""
    print(f"\n=== 测试 Base64 OCR: {image_path} ===")
    
    if not Path(image_path).exists():
        print(f"错误: 图片文件不存在: {image_path}")
        return False
    
    # 读取图片并转换为 Base64
    with open(image_path, 'rb') as f:
        image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
    
    data = {
        'image_base64': image_base64,
        'prompt': '<image>\nFree OCR.',
        'base_size': 1024,
        'image_size': 640,
        'crop_mode': True
    }
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/ocr/base64", data=data)
    elapsed_time = time.time() - start_time
    
    print(f"状态码: {response.status_code}")
    print(f"请求耗时: {elapsed_time:.2f}s")
    
    if response.status_code == 200:
        result = response.json()
        print(f"成功: {result['success']}")
        if result['success']:
            print(f"处理时间: {result['processing_time']:.2f}s")
            print(f"识别文本长度: {len(result['text'])} 字符")
            print(f"识别文本预览:\n{result['text'][:500]}...")
        else:
            print(f"错误: {result['error']}")
        return result['success']
    else:
        print(f"请求失败: {response.text}")
        return False


def test_batch_ocr(image_paths: list):
    """测试批量 OCR"""
    print(f"\n=== 测试批量 OCR: {len(image_paths)} 张图片 ===")
    
    files = []
    for image_path in image_paths:
        if Path(image_path).exists():
            files.append(
                ('files', (Path(image_path).name, open(image_path, 'rb'), 'image/jpeg'))
            )
    
    if not files:
        print("错误: 没有有效的图片文件")
        return False
    
    data = {
        'prompt': '<image>\n<|grounding|>Convert the document to markdown.',
        'base_size': 1024,
        'image_size': 640,
        'crop_mode': True
    }
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/ocr/batch", files=files, data=data)
    elapsed_time = time.time() - start_time
    
    # 关闭文件
    for _, file_tuple in files:
        file_tuple[1].close()
    
    print(f"状态码: {response.status_code}")
    print(f"请求耗时: {elapsed_time:.2f}s")
    
    if response.status_code == 200:
        results = response.json()
        success_count = sum(1 for r in results if r['success'])
        print(f"成功: {success_count}/{len(results)}")
        for i, result in enumerate(results):
            if result['success']:
                print(f"  图片 {i+1}: 识别文本长度 {len(result['text'])} 字符")
            else:
                print(f"  图片 {i+1}: 失败 - {result['error']}")
        return success_count == len(results)
    else:
        print(f"请求失败: {response.text}")
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("DeepSeek-OCR API 测试")
    print("=" * 60)
    
    # 测试基础接口
    test_root()
    test_health()
    
    # 等待服务完全启动
    print("\n等待服务完全启动...")
    time.sleep(2)
    
    # 测试 OCR 功能（需要提供测试图片）
    test_image = "test_image.jpg"  # 替换为实际的测试图片路径
    
    if Path(test_image).exists():
        test_ocr_image(test_image)
        test_ocr_base64(test_image)
    else:
        print(f"\n提示: 请提供测试图片 '{test_image}' 以测试 OCR 功能")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()

