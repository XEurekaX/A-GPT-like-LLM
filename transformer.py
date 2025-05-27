from selfattention import MultiHeadAttention
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # 计算最后一个维度的均值
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 计算最后一个维度的方差
        norm_x = (x - mean) / torch.sqrt(var + self.eps)  # 归一化
        return self.scale * norm_x + self.shift  # 缩放和平移


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))  


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  
            GELU(),  
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  
        )

    def forward(self, x):
        return self.layers(x) 



class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])  # 词嵌入层
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # 位置嵌入层
        self.drop_emb = nn.Dropout(cfg["drop_rate"])  # Dropout

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])  # Transformer块堆叠

        self.final_norm = LayerNorm(cfg["emb_dim"])  # 最终归一化层
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)  # 输出层

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)  # 获取词嵌入
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # 获取位置嵌入
        x = tok_embeds + pos_embeds  # 将词嵌入和位置嵌入相加
        x = self.drop_emb(x)  # Dropout
        x = self.trf_blocks(x)  # 通过Transformer块
        x = self.final_norm(x)  # 最终归一化
        logits = self.out_head(x)  # 输出对数概率
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx是当前上下文中索引的(B, T)数组
    for _ in range(max_new_tokens):

        # 如果当前上下文超过支持的最大长度，则进行裁剪
        # 例如，如果LLM只支持5个token，而上下文长度是10
        # 那么只使用最后5个token作为上下文
        idx_cond = idx[:, -context_size:]

        # 获取预测结果
        with torch.no_grad():
            logits = model(idx_cond)

        # 只关注最后一个时间步
        # (batch, n_token, vocab_size)变为(batch, vocab_size)
        logits = logits[:, -1, :]

        # 获取logits值最高的词汇表索引
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # 将采样到的索引添加到运行序列中
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 线性层，扩展维度
            nn.GELU(approximate="tanh"),  # 近似GELU激活函数
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 线性层，恢复原始维度
        )

    def forward(self, x):
        return self.layers(x)  # 前向传播


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # 残差连接：注意力块
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # 注意力机制
        x = self.drop_shortcut(x)  # Dropout
        x = x + shortcut  # 残差连接

        # 残差连接：前馈网络块
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)  # 前馈网络
        x = self.drop_shortcut(x)  # Dropout
        x = x + shortcut  # 残差连接

        return x