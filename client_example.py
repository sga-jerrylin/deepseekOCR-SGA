"""
DeepSeek-OCR API 客户端示例
演示如何调用 API 进行 OCR 识别
"""

import requests
import base64
import json
from pathlib import Path
from typing import Optional, List


class DeepSeekOCRClient:
    """DeepSeek-OCR API 客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        初始化客户端
        
        Args:
            base_url: API 服务地址
        """
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> dict:
        """
        健康检查
        
        Returns:
            健康状态信息
        """
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def ocr_image(
        self,
        image_path: str,
        prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
        base_size: int = 1024,
        image_size: int = 640,
        crop_mode: bool = True,
        save_results: bool = False
    ) -> dict:
        """
        对图片进行 OCR 识别
        
        Args:
            image_path: 图片文件路径
            prompt: OCR 提示词
            base_size: 基础尺寸
            image_size: 图片尺寸
            crop_mode: 是否裁剪模式
            save_results: 是否保存结果
            
        Returns:
            OCR 识别结果
        """
        with open(image_path, 'rb') as f:
            files = {'file': (Path(image_path).name, f, 'image/jpeg')}
            data = {
                'prompt': prompt,
                'base_size': base_size,
                'image_size': image_size,
                'crop_mode': crop_mode,
                'save_results': save_results
            }
            
            response = requests.post(
                f"{self.base_url}/ocr/image",
                files=files,
                data=data
            )
            response.raise_for_status()
            return response.json()
    
    def ocr_base64(
        self,
        image_path: str,
        prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
        base_size: int = 1024,
        image_size: int = 640,
        crop_mode: bool = True
    ) -> dict:
        """
        对 Base64 编码的图片进行 OCR 识别
        
        Args:
            image_path: 图片文件路径
            prompt: OCR 提示词
            base_size: 基础尺寸
            image_size: 图片尺寸
            crop_mode: 是否裁剪模式
            
        Returns:
            OCR 识别结果
        """
        with open(image_path, 'rb') as f:
            image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        data = {
            'image_base64': image_base64,
            'prompt': prompt,
            'base_size': base_size,
            'image_size': image_size,
            'crop_mode': crop_mode
        }
        
        response = requests.post(
            f"{self.base_url}/ocr/base64",
            data=data
        )
        response.raise_for_status()
        return response.json()
    
    def ocr_batch(
        self,
        image_paths: List[str],
        prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
        base_size: int = 1024,
        image_size: int = 640,
        crop_mode: bool = True
    ) -> List[dict]:
        """
        批量处理多个图片
        
        Args:
            image_paths: 图片文件路径列表
            prompt: OCR 提示词
            base_size: 基础尺寸
            image_size: 图片尺寸
            crop_mode: 是否裁剪模式
            
        Returns:
            OCR 识别结果列表
        """
        files = []
        for image_path in image_paths:
            files.append(
                ('files', (Path(image_path).name, open(image_path, 'rb'), 'image/jpeg'))
            )
        
        data = {
            'prompt': prompt,
            'base_size': base_size,
            'image_size': image_size,
            'crop_mode': crop_mode
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/ocr/batch",
                files=files,
                data=data
            )
            response.raise_for_status()
            return response.json()
        finally:
            # 关闭文件
            for _, file_tuple in files:
                file_tuple[1].close()


# ============= 使用示例 =============

def example_1_basic_ocr():
    """示例 1: 基础 OCR 识别"""
    print("\n" + "="*60)
    print("示例 1: 基础 OCR 识别")
    print("="*60)
    
    client = DeepSeekOCRClient()
    
    # 检查服务状态
    try:
        health = client.health_check()
        print(f"服务状态: {health['status']}")
        print(f"模型已加载: {health['model_loaded']}")
        print(f"设备: {health['device']}")
    except Exception as e:
        print(f"错误: 无法连接到服务 - {e}")
        return
    
    # OCR 识别
    image_path = "test_image.jpg"  # 替换为实际图片路径
    
    if not Path(image_path).exists():
        print(f"提示: 请提供测试图片 '{image_path}'")
        return
    
    try:
        result = client.ocr_image(image_path)
        
        if result['success']:
            print(f"\n识别成功!")
            print(f"处理时间: {result['processing_time']:.2f}s")
            print(f"识别文本长度: {len(result['text'])} 字符")
            print(f"\n识别结果:\n{result['text'][:500]}...")
        else:
            print(f"识别失败: {result['error']}")
    except Exception as e:
        print(f"错误: {e}")


def example_2_different_prompts():
    """示例 2: 使用不同的提示词"""
    print("\n" + "="*60)
    print("示例 2: 使用不同的提示词")
    print("="*60)
    
    client = DeepSeekOCRClient()
    image_path = "test_image.jpg"
    
    if not Path(image_path).exists():
        print(f"提示: 请提供测试图片 '{image_path}'")
        return
    
    # 不同的提示词
    prompts = {
        "文档转 Markdown": "<image>\n<|grounding|>Convert the document to markdown.",
        "通用 OCR": "<image>\n<|grounding|>OCR this image.",
        "无布局识别": "<image>\nFree OCR.",
        "详细描述": "<image>\nDescribe this image in detail."
    }
    
    for name, prompt in prompts.items():
        print(f"\n--- {name} ---")
        try:
            result = client.ocr_image(image_path, prompt=prompt)
            if result['success']:
                print(f"识别文本: {result['text'][:200]}...")
        except Exception as e:
            print(f"错误: {e}")


def example_3_batch_processing():
    """示例 3: 批量处理"""
    print("\n" + "="*60)
    print("示例 3: 批量处理")
    print("="*60)
    
    client = DeepSeekOCRClient()
    
    # 批量处理多张图片
    image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
    
    # 过滤存在的图片
    existing_images = [p for p in image_paths if Path(p).exists()]
    
    if not existing_images:
        print("提示: 请提供测试图片")
        return
    
    try:
        results = client.ocr_batch(existing_images)
        
        for i, result in enumerate(results):
            print(f"\n图片 {i+1}: {existing_images[i]}")
            if result['success']:
                print(f"  处理时间: {result['processing_time']:.2f}s")
                print(f"  文本长度: {len(result['text'])} 字符")
                print(f"  预览: {result['text'][:100]}...")
            else:
                print(f"  失败: {result['error']}")
    except Exception as e:
        print(f"错误: {e}")


def example_4_base64_encoding():
    """示例 4: Base64 编码方式"""
    print("\n" + "="*60)
    print("示例 4: Base64 编码方式")
    print("="*60)
    
    client = DeepSeekOCRClient()
    image_path = "test_image.jpg"
    
    if not Path(image_path).exists():
        print(f"提示: 请提供测试图片 '{image_path}'")
        return
    
    try:
        result = client.ocr_base64(image_path)
        
        if result['success']:
            print(f"识别成功!")
            print(f"处理时间: {result['processing_time']:.2f}s")
            print(f"识别文本: {result['text'][:200]}...")
        else:
            print(f"识别失败: {result['error']}")
    except Exception as e:
        print(f"错误: {e}")


def main():
    """主函数"""
    print("="*60)
    print("DeepSeek-OCR API 客户端示例")
    print("="*60)
    
    # 运行示例
    example_1_basic_ocr()
    # example_2_different_prompts()
    # example_3_batch_processing()
    # example_4_base64_encoding()
    
    print("\n" + "="*60)
    print("示例完成")
    print("="*60)


if __name__ == "__main__":
    main()

