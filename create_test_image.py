"""
创建一个简单的测试图片
"""

from PIL import Image, ImageDraw, ImageFont

# 创建一个白色背景的图片
img = Image.new('RGB', (800, 600), color='white')
draw = ImageDraw.Draw(img)

# 添加一些文本
try:
    # 尝试使用系统字体
    font = ImageFont.truetype("arial.ttf", 40)
except:
    # 如果找不到字体，使用默认字体
    font = ImageFont.load_default()

# 绘制文本
text = """DeepSeek-OCR Test

This is a test document for OCR.

Features:
- High accuracy
- Fast processing
- GPU acceleration

Date: 2025-10-30
"""

draw.text((50, 50), text, fill='black', font=font)

# 保存图片
img.save('test_image.jpg')
print("✅ 测试图片已创建: test_image.jpg")

