"""
使用官方示例代码直接测试 DeepSeek-OCR
"""
from transformers import AutoModel, AutoTokenizer
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# 禁用 JIT 编译
os.environ["PYTORCH_JIT"] = "0"
os.environ["PYTORCH_NVFUSER_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

model_name = 'deepseek-ai/DeepSeek-OCR'

print("🚀 开始加载模型...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name, 
    attn_implementation='eager',  # 使用 eager 模式
    trust_remote_code=True, 
    use_safetensors=True
)

print("📦 模型加载完成,正在移到 GPU...")
model = model.eval().cuda().to(torch.bfloat16)

print("✅ 模型准备完成! (GPU 模式)")

prompt = "<image>\n<|grounding|>Convert the document to markdown."
image_file = 'test_image.jpg'
output_path = './test_output'

print(f"\n📄 开始 OCR 识别: {image_file}")
print(f"💡 提示词: {prompt}")

try:
    res = model.infer(
        tokenizer, 
        prompt=prompt, 
        image_file=image_file, 
        output_path=output_path, 
        base_size=1024, 
        image_size=640, 
        crop_mode=True, 
        save_results=True, 
        test_compress=True,
        eval_mode=True  # 使用 eval 模式,不使用 streamer
    )
    
    print("\n✅ OCR 识别完成!")
    print(f"\n📝 识别结果:\n{res}")
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()

