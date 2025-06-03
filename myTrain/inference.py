import torch
import tiktoken  # OpenAI的分词库
from transformer import GPTModel
# 使用项目中已有的生成函数
from train_and_gen_class import generate, text_to_token_ids, token_ids_to_text

# 配置参数（需要根据你的模型配置调整）
cfg = {
    "vocab_size": 50257,     # GPT-2词汇表大小
    "emb_dim": 768,          # 嵌入维度
    "n_layers": 12,          # Transformer块的数量
    "n_heads": 12,           # 注意力头的数量
    "context_length": 256,  # 上下文长度
    "drop_rate": 0.1,        # Dropout率
    "qkv_bias": False        # 是否使用QKV偏置
}


# 创建模型实例
device = torch.device("cpu")
model = GPTModel(cfg).to(device)

# 加载训练好的权重
try:
    model.load_state_dict(torch.load("models/model_epoch_10.pth", map_location=device),strict=False)
    print("成功加载模型权重")
except FileNotFoundError:
    print("错误：模型文件未找到，请检查路径是否正确")
    print("提示：请将训练好的模型权重文件放在models/目录下")
    print("提示：确保模型文件名与你尝试加载的文件名一致")
    exit(1)
except KeyError as e:
    print(f"加载模型失败：缺少必要的配置参数 {e}")
    print("请检查模型配置是否与训练时使用的配置匹配")
    exit(1)
except Exception as e:
    print(f"加载模型时发生未知错误: {e}")
    exit(1)

# 设置为评估模式
model.eval()

# 文本生成函数
def generate_text(model, prompt, max_new_tokens=50):

    tokenizer = tiktoken.get_encoding("gpt2")  # 使用与训练时相同的分词器
    
    # 使用项目中定义的分词函数Once upon a time
    encoded = text_to_token_ids(prompt, tokenizer)  # 传递分词器参数
    
    # 修复：确保encoded是张量并在设备上
    if not isinstance(encoded, torch.Tensor):
        encoded = torch.tensor(encoded).unsqueeze(0)  # 添加批次维度
    idx = encoded.to(device)
    
    # 修复：确保模型在正确设备上
    model.to(device)
    
    # 生成文本
    with torch.no_grad():
        idx_generated = generate(
            model=model,
            idx=idx,
            max_new_tokens=max_new_tokens,
            context_size=cfg["context_length"],
            temperature=1,  # 添加温度参数控制生成多样性
            top_k=5,         # 添加top_k采样
            eos_id=None       # 结束符ID（如果有的话）
        )
    
    # 解码生成的token
    decoded = token_ids_to_text(idx_generated, tokenizer)  # 传递分词器参数
    return decoded

# 示例用法
if __name__ == "__main__":
    input_text = "man what can i"
    print(f"生成文本基于提示: {input_text}")
    
    # 修复：使用与generate函数签名匹配的参数
    generated_text = generate_text(
        model=model,
        prompt=input_text,
        max_new_tokens=256
    )
    print("\n生成结果:")
    print(generated_text)