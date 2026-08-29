"""
python 3-bert-adapter.py
export OMP_NUM_THREADS=4
预训练 HyperNetwork：输入 = (朝代特征 + 语境信息) -> 输出 = dynasty adapter 向量
训练完成后：为每个 dynasty 生成一个固定 adapter（聚合/平均），保存到 pt 文件
"""

import os
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from transformers import BertTokenizer, BertModel


# -------------------------
# Config
# -------------------------
@dataclass
class HyperAdapterConfig:
    model_name: str = "bert-base-chinese"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 预训练超网络用到的少量数据
    max_length: int = 256
    batch_size: int = 16
    lr: float = 5e-4
    epochs: int = 30

    # 每个 dynasty 采样多少条做 hypernetwork 训练
    samples_per_dynasty: int = 200

    # 超网络结构
    dynasty_emb_dim: int = 64
    hyper_hidden_dim: int = 256

    # adapter 维度：这里直接生成 hidden_size 维向量（作为“特征偏置/辅助向量”）
    # 如果想生成真正的 Adapter(Down/Up 的权重矩阵)，维度会非常大，不适合这种“先训再保存”的轻量方案
    # （要做也可以，但工程复杂且文件很大）
    adapter_dim: Optional[int] = None  # None 表示 = bert_hidden_size

    # 输出保存路径
    save_path: str = "dynasty_adapters.pt"

    seed: int = 42


# -------------------------
# Dataset for hypernetwork pretrain
# -------------------------
class HyperPretrainDataset(Dataset):
    """
    预训练数据：只需要 text + word（作为 pair 输入） + dynasty(dynasty)
    训练目标：用 (dynasty adapter + CLS) 去预测 label（弱监督），迫使 adapter 学到可用信息。
    """

    def __init__(self, data: List[Dict], tokenizer: BertTokenizer, max_length: int,
                 label_to_idx: Dict[str, int]):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        text = str(item.get("text", ""))
        word = str(item.get("word", ""))
        dynasty = str(item.get("dynasty", "unknown"))
        label = str(item.get("label", "unknown"))

        encoding = self.tokenizer(
            text,
            word,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "dynasty": dynasty,
            "label_idx": self.label_to_idx.get(label, -1),
        }


def collate_hyper(batch: List[Dict]):
    input_ids = torch.stack([x["input_ids"] for x in batch], dim=0)
    attention_mask = torch.stack([x["attention_mask"] for x in batch], dim=0)
    dynasties = [x["dynasty"] for x in batch]
    label_idx = torch.tensor([x["label_idx"] for x in batch], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "dynasties": dynasties,
        "label_idx": label_idx,
    }


# -------------------------
# HyperNetwork
# -------------------------
class DynastyContextHyperNet(nn.Module):
    """
    输入: dynasty embedding + context CLS
    输出: adapter 向量 (adapter_dim)
    """

    def __init__(self, num_dynasties: int, dynasty_emb_dim: int,
                 context_dim: int, hyper_hidden_dim: int, adapter_dim: int):
        super().__init__()
        self.dynasty_emb = nn.Embedding(num_dynasties, dynasty_emb_dim)

        in_dim = dynasty_emb_dim + context_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(hyper_hidden_dim, hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(hyper_hidden_dim, adapter_dim),
        )

    def forward(self, dynasty_idx: torch.LongTensor, context_vec: torch.Tensor) -> torch.Tensor:
        """
        dynasty_idx: [B]
        context_vec: [B, H]
        return: [B, adapter_dim]
        """
        d = self.dynasty_emb(dynasty_idx)  # [B, E]
        x = torch.cat([d, context_vec], dim=-1)
        a = self.mlp(x)
        return a


# -------------------------
# Pretrain objective head
# -------------------------
class WordIdPredictor(nn.Module):
    """
    用 (CLS + adapter) 预测 label（弱监督），让 adapter 学到“与词相关的时代/语境偏置”
    """
    def __init__(self, hidden_size: int, adapter_dim: int, num_labels: int):
        super().__init__()
        self.proj = nn.Linear(adapter_dim, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, cls: torch.Tensor, adapter: torch.Tensor) -> torch.Tensor:
        # cls: [B,H], adapter:[B,Ad]
        h = cls + self.proj(adapter)
        return self.classifier(h)


# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def stratified_sample_by_dynasty(data: List[Dict], samples_per_dynasty: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    groups = defaultdict(list)
    for item in data:
        groups[str(item.get("dynasty", "unknown"))].append(item)

    sampled = []
    for dynasty, items in groups.items():
        if len(items) <= samples_per_dynasty:
            sampled.extend(items)
        else:
            sampled.extend(rng.sample(items, samples_per_dynasty))
    rng.shuffle(sampled)
    return sampled


def build_label_vocab(data: List[Dict]) -> Dict[str, int]:
    labels = sorted({str(x.get("label", "unknown")) for x in data})
    return {w: i for i, w in enumerate(labels)}


def build_dynasty_vocab(data: List[Dict]) -> Dict[str, int]:
    dynasties = sorted({str(x.get("dynasty", "unknown")) for x in data})
    return {d: i for i, d in enumerate(dynasties)}


# -------------------------
# Main pretrain
# -------------------------
def pretrain_and_save_adapters(train_json: str, config: HyperAdapterConfig):
    set_seed(config.seed)
    device = torch.device(config.device)

    data_all = load_json(train_json)
    data = stratified_sample_by_dynasty(data_all, config.samples_per_dynasty, config.seed)

    dynasty_to_idx = build_dynasty_vocab(data_all)  # 用全量数据统计 dynasty 集合更稳
    label_to_idx = build_label_vocab(data_all)

    tokenizer = BertTokenizer.from_pretrained(config.model_name)
    bert = BertModel.from_pretrained(config.model_name).to(device)
    bert.eval()  # 语境向量提取器一般冻结（减少不稳定性）
    for p in bert.parameters():
        p.requires_grad = False

    hidden_size = bert.config.hidden_size
    adapter_dim = config.adapter_dim or hidden_size

    hypernet = DynastyContextHyperNet(
        num_dynasties=len(dynasty_to_idx),
        dynasty_emb_dim=config.dynasty_emb_dim,
        context_dim=hidden_size,
        hyper_hidden_dim=config.hyper_hidden_dim,
        adapter_dim=adapter_dim,
    ).to(device)

    predictor = WordIdPredictor(
        hidden_size=hidden_size,
        adapter_dim=adapter_dim,
        num_labels=len(label_to_idx),
    ).to(device)

    dataset = HyperPretrainDataset(
        data=data,
        tokenizer=tokenizer,
        max_length=config.max_length,
        label_to_idx=label_to_idx,
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_hyper)

    optim = AdamW(list(hypernet.parameters()) + list(predictor.parameters()), lr=config.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    hypernet.train()
    predictor.train()

    for epoch in range(config.epochs):
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"[HyperPretrain] epoch {epoch+1}/{config.epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            dynasties = batch["dynasties"]
            label_idx = batch["label_idx"].to(device)

            with torch.no_grad():
                out = bert(input_ids=input_ids, attention_mask=attention_mask)
                cls = out.last_hidden_state[:, 0, :]  # [B,H]

            dynasty_idx = torch.tensor([dynasty_to_idx.get(d, dynasty_to_idx["unknown"]) if "unknown" in dynasty_to_idx else dynasty_to_idx.get(d, 0)
                                        for d in dynasties], dtype=torch.long, device=device)

            adapter = hypernet(dynasty_idx, cls)  # [B,Ad]
            logits = predictor(cls, adapter)      # [B, num_labels]

            loss = loss_fn(logits, label_idx)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(hypernet.parameters()) + list(predictor.parameters()), 1.0)
            optim.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = total_loss / max(1, len(loader))
        print(f"epoch={epoch+1} avg_loss={avg:.4f}")

    # --------- 生成每个 dynasty 的固定 adapter（聚合平均）---------
    hypernet.eval()
    dynasty_adapters = {}
    dynasty_counts = defaultdict(int)
    dynasty_sums = defaultdict(lambda: torch.zeros(adapter_dim, device=device))

    # 用全量数据（或更大采样）来做聚合更稳，这里用 data_all（但也可用 data）
    agg_dataset = HyperPretrainDataset(
        data=data_all,
        tokenizer=tokenizer,
        max_length=config.max_length,
        label_to_idx=label_to_idx,
    )
    agg_loader = DataLoader(agg_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_hyper)

    with torch.no_grad():
        for batch in tqdm(agg_loader, desc="[Aggregate adapters]"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            dynasties = batch["dynasties"]

            out = bert(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]

            dynasty_idx = torch.tensor([dynasty_to_idx.get(d, dynasty_to_idx["unknown"]) if "unknown" in dynasty_to_idx else dynasty_to_idx.get(d, 0)
                                        for d in dynasties], dtype=torch.long, device=device)

            adapter_batch = hypernet(dynasty_idx, cls)  # [B,Ad]

            for i, d in enumerate(dynasties):
                d = str(d)
                dynasty_sums[d] += adapter_batch[i]
                dynasty_counts[d] += 1

    for d, s in dynasty_sums.items():
        c = max(1, dynasty_counts[d])
        dynasty_adapters[d] = (s / c).detach().cpu()  # [Ad]

    payload = {
        "model_name": config.model_name,
        "adapter_dim": adapter_dim,
        "dynasty_to_idx": dynasty_to_idx,
        "label_to_idx": label_to_idx,
        "dynasty_adapters": dynasty_adapters,  # Dict[str, Tensor(adapter_dim)]
        "hyper_config": config.__dict__,
    }

    os.makedirs(os.path.dirname(config.save_path) or ".", exist_ok=True)
    torch.save(payload, config.save_path)
    print(f"✓ Saved dynasty adapters to: {config.save_path}")
    print(f"✓ dynasties: {len(dynasty_adapters)} | adapter_dim={adapter_dim}")


if __name__ == "__main__":
    # 你可以替换成你的训练集路径
    TRAIN_JSON = "/root/autodl-tmp/atd/data/人工标注/最终结果/train.json"
    cfg = HyperAdapterConfig(
        model_name="/root/autodl-tmp/models/bert-base-chinese",
        save_path="/root/autodl-tmp/atd/data/结果/PLM/dynasty_adapters.pt",
        samples_per_dynasty=200,
        epochs=60,
    )
    pretrain_and_save_adapters(TRAIN_JSON, cfg)