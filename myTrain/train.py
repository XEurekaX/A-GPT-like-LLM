import json
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
from torch.nn.utils.rnn import pad_sequence

from transformer import GPTModel
from train_and_gen_class import train_model

class JSONLDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=256):
        self.data = []
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                # 拼接input和content作为训练样本
                full_text = item['input'] + ' ' + item['content']
                token_ids = tokenizer.encode(full_text)
                self.data.append(token_ids)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        token_ids = self.data[idx]
        # 截断处理
        token_ids = token_ids[:self.max_length]
        # 输入和目标序列
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(token_ids[1:], dtype=torch.long)
        return input_ids, target_ids

def create_jsonl_dataloader(file_path, batch_size=4, max_length=256):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = JSONLDataset(file_path, tokenizer, max_length)

    def collate_fn(batch):
        input_ids_list = [item[0] for item in batch]
        target_ids_list = [item[1] for item in batch]
        input_ids_padded = pad_sequence(input_ids_list, batch_first=True)
        target_ids_padded = pad_sequence(target_ids_list, batch_first=True)
        return input_ids_padded, target_ids_padded

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=collate_fn
    )

def main():
    # 配置参数
    GPT_CONFIG = {
        "vocab_size": 50257,     # GPT-2词汇表大小
        "context_length": 256,   # 上下文长度
        "emb_dim": 768,          # 嵌入维度
        "n_heads": 12,           # 注意力头数量
        "n_layers": 12,          # 层数
        "drop_rate": 0.1,        # dropout率
        "qkv_bias": False        # 查询-键-值偏置
    }

    # 创建数据加载器

    train_loader = create_jsonl_dataloader("data/train.jsonl", batch_size=8)
    val_loader = create_jsonl_dataloader("data/val.jsonl", batch_size=8)
    print(f"训练集 batch 数量: {len(train_loader)}")
    print(f"验证集 batch 数量: {len(val_loader)}")

    print(f"训练集样本总数: {len(train_loader.dataset)}")
    print(f"验证集样本总数: {len(val_loader.dataset)}")

    # 初始化模型
    model = GPTModel(GPT_CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

    # 训练模型
    train_model(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=100, eval_freq=10, eval_iter=5,
        start_context="如何评价贴吧", tokenizer=tiktoken.get_encoding("gpt2")
    )
    # 保存模型
    torch.save(model.state_dict(), "jsonl_gpt_model.pth")

if __name__ == "__main__":
    main()