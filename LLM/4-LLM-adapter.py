import os
import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------
# Config
# -------------------------
@dataclass
class HyperAdapterConfig:
    model_name: str = "Qwen/Qwen3-1.5B-Instruct"  # LLM 原始名称
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 预训练超网络用到的少量数据
    max_length: int = 256
    # 32
    batch_size: int = 8
    lr: float = 5e-4
    epochs: int = 30

    # 每个 dynasty 采样多少条做 hypernetwork 训练
    samples_per_dynasty: int = 100

    # 超网络结构
    dynasty_emb_dim: int = 64
    hyper_hidden_dim: int = 256

    # adapter 维度：默认等于 context_dim (LLM 隐层维度)
    adapter_dim: Optional[int] = None

    # 输出保存路径
    save_path: str = "dynasty_adapters.pt"

    seed: int = 42

    # 新增：LLM encoder 相关
    pooling: str = "mean"  # "mean" or "cls"
    use_xinference: bool = False
    xinference_base_url: str = "http://127.0.0.1:9997"  # 对话模式走 HTTP


# -------------------------
# Dataset (raw text only)
# -------------------------
class HyperPretrainDataset(Dataset):
    """
    预训练数据：只需要 text + word (作为 pair 输入) + dynasty(dynasty)
    训练目标：用 (dynasty adapter + context_vec) 去预测 label (弱监督)
    """

    def __init__(self, data: List[Dict], max_length: int,
                 label_to_idx: Dict[str, int]):
        self.data = data
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

        return {
            "text": text,
            "word": word,
            "dynasty": dynasty,
            "label_idx": self.label_to_idx.get(label, -1),
        }


def collate_hyper(batch: List[Dict]):
    texts = [x["text"] for x in batch]
    words = [x["word"] for x in batch]
    dynasties = [x["dynasty"] for x in batch]
    label_idx = torch.tensor([x["label_idx"] for x in batch], dtype=torch.long)
    return {
        "texts": texts,
        "words": words,
        "dynasties": dynasties,
        "label_idx": label_idx,
    }


# -------------------------
# HyperNetwork
# -------------------------
class DynastyContextHyperNet(nn.Module):
    """
    输入: dynasty embedding + context_vec
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
        d = self.dynasty_emb(dynasty_idx)  # [B, E]
        x = torch.cat([d, context_vec], dim=-1)
        a = self.mlp(x)
        return a


# -------------------------
# Pretrain objective head
# -------------------------
class LabelPredictor(nn.Module):
    """
    用 (context + adapter) 预测 label (弱监督)
    """
    def __init__(self, hidden_size: int, adapter_dim: int, num_labels: int):
        super().__init__()
        self.proj = nn.Linear(adapter_dim, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, context_vec: torch.Tensor, adapter: torch.Tensor) -> torch.Tensor:
        h = context_vec + self.proj(adapter)
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
# LLM Encoder (Local HF or Xinference)
# -------------------------
class LLMEncoder:
    def __init__(self, model_name: str, max_length: int, pooling: str = "mean",
                 use_xinference: bool = False, xinference_base_url: str = "http://127.0.0.1:9997"):
        self.model_name = model_name
        self.max_length = max_length
        self.pooling = pooling
        self.use_xinference = use_xinference
        self.xinference_base_url = xinference_base_url

        if not use_xinference:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            self.model.eval()
            self.hidden_size = self.model.config.hidden_size
        else:
            # xinference HTTP 模式：这里假设提供 embeddings/或者 chat 接口返回 hidden_states
            # 如果你的 xinference 没有 embeddings，建议在服务端开启 embeddings
            import requests
            self.requests = requests

            # 尝试通过一次 embeddings 获取维度 (如果可用)
            # 如果不可用，请手动设置 hidden_size
            self.hidden_size = None

    def _mean_pool(self, hidden_states, attention_mask):
        # hidden_states: [B, T, H]
        mask = attention_mask.unsqueeze(-1).float()
        summed = torch.sum(hidden_states * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-6)
        return summed / counts

    @torch.no_grad()
    def encode_batch(self, texts: List[str], words: List[str]) -> torch.Tensor:
        # 拼接成 pair 输入 (保持和原始流程一致)
        inputs = [f"{t}\n{w}" for t, w in zip(texts, words)]

        if not self.use_xinference:
            tok = self.tokenizer(
                inputs,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            tok = {k: v.to(self.model.device) for k, v in tok.items()}
            out = self.model(**tok, output_hidden_states=True)

            # 最后一层 hidden states
            hs = out.hidden_states[-1]  # [B, T, H]

            if self.pooling == "cls":
                # 取第一个 token
                vec = hs[:, 0, :]
            else:
                vec = self._mean_pool(hs, tok["attention_mask"])

            return vec.detach().cpu()
        else:
            # xinference 模式：
            # 默认走 /v1/embeddings
            url = f"{self.xinference_base_url}/v1/embeddings"
            payload = {
                "model": self.model_name,
                "input": inputs
            }
            resp = self.requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            # 适配标准 OpenAI embeddings 格式
            embeddings = [item["embedding"] for item in data["data"]]
            vec = torch.tensor(embeddings, dtype=torch.float32)
            if self.hidden_size is None:
                self.hidden_size = vec.shape[-1]
            return vec


# -------------------------
# Main pretrain
# -------------------------
def pretrain_and_save_adapters(train_json: str, config: HyperAdapterConfig):
    set_seed(config.seed)
    device = torch.device(config.device)

    data_all = load_json(train_json)
    data = stratified_sample_by_dynasty(data_all, config.samples_per_dynasty, config.seed)

    dynasty_to_idx = build_dynasty_vocab(data_all)
    label_to_idx = build_label_vocab(data_all)

    encoder = LLMEncoder(
        model_name=config.model_name,
        max_length=config.max_length,
        pooling=config.pooling,
        use_xinference=config.use_xinference,
        xinference_base_url=config.xinference_base_url
    )

    # 如果是 HF，本地可直接拿 hidden_size；xinference embeddings 会在首次 encode 时补齐
    if encoder.hidden_size is None:
        # 触发一次 dummy encode，得到 hidden_size
        _ = encoder.encode_batch(["测试"], ["测试"])
    hidden_size = encoder.hidden_size
    adapter_dim = config.adapter_dim or hidden_size

    hypernet = DynastyContextHyperNet(
        num_dynasties=len(dynasty_to_idx),
        dynasty_emb_dim=config.dynasty_emb_dim,
        context_dim=hidden_size,
        hyper_hidden_dim=config.hyper_hidden_dim,
        adapter_dim=adapter_dim,
    ).to(device)

    predictor = LabelPredictor(
        hidden_size=hidden_size,
        adapter_dim=adapter_dim,
        num_labels=len(label_to_idx),
    ).to(device)

    dataset = HyperPretrainDataset(
        data=data,
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
            texts = batch["texts"]
            words = batch["words"]
            dynasties = batch["dynasties"]
            label_idx = batch["label_idx"].to(device)

            with torch.no_grad():
                ctx_vec = encoder.encode_batch(texts, words).to(device)  # [B, H]

            dynasty_idx = torch.tensor(
                [dynasty_to_idx.get(d, dynasty_to_idx["unknown"]) if "unknown" in dynasty_to_idx else dynasty_to_idx.get(d, 0)
                 for d in dynasties],
                dtype=torch.long,
                device=device
            )

            adapter = hypernet(dynasty_idx, ctx_vec)  # [B, Ad]
            logits = predictor(ctx_vec, adapter)      # [B, num_labels]

            loss = loss_fn(logits, label_idx)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(hypernet.parameters()) + list(predictor.parameters()), 1.0)
            optim.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = total_loss / max(1, len(loader))
        print(f"epoch={epoch+1} avg_loss={avg:.4f}")

    # --------- 生成每个 dynasty 的固定 adapter (聚合平均)---------
    hypernet.eval()
    dynasty_adapters = {}
    dynasty_counts = defaultdict(int)
    dynasty_sums = defaultdict(lambda: torch.zeros(adapter_dim, device=device))

    agg_dataset = HyperPretrainDataset(
        data=data_all,
        max_length=config.max_length,
        label_to_idx=label_to_idx,
    )
    agg_loader = DataLoader(agg_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_hyper)

    with torch.no_grad():
        for batch in tqdm(agg_loader, desc="[Aggregate adapters]"):
            texts = batch["texts"]
            words = batch["words"]
            dynasties = batch["dynasties"]

            ctx_vec = encoder.encode_batch(texts, words).to(device)
            dynasty_idx = torch.tensor(
                [dynasty_to_idx.get(d, dynasty_to_idx["unknown"]) if "unknown" in dynasty_to_idx else dynasty_to_idx.get(d, 0)
                 for d in dynasties],
                dtype=torch.long,
                device=device
            )

            adapter_batch = hypernet(dynasty_idx, ctx_vec)  # [B,Ad]

            for i, d in enumerate(dynasties):
                d = str(d)
                dynasty_sums[d] += adapter_batch[i]
                dynasty_counts[d] += 1

    for d, s in dynasty_sums.items():
        c = max(1, dynasty_counts[d])
        dynasty_adapters[d] = (s / c).detach().cpu()

    payload = {
        "model_name": config.model_name,
        "adapter_dim": adapter_dim,
        "dynasty_to_idx": dynasty_to_idx,
        "label_to_idx": label_to_idx,
        "dynasty_adapters": dynasty_adapters,
        "hyper_config": config.__dict__,
    }

    os.makedirs(os.path.dirname(config.save_path) or ".", exist_ok=True)
    torch.save(payload, config.save_path)
    print(f"✓ Saved dynasty adapters to: {config.save_path}")
    print(f"✓ dynasties: {len(dynasty_adapters)} | adapter_dim={adapter_dim}")


if __name__ == "__main__":
    TRAIN_JSON = str(Path(__file__).resolve().parents[1] / "data" / "wsd_train.json")
    cfg = HyperAdapterConfig(
        model_name="/root/autodl-tmp/models/Qwen3-1.5B-Instruct",  # 可换成你列出的任意模型
        save_path="/root/autodl-tmp/atd/data/结果/LLM/dynasty_adapters.pt",
        samples_per_dynasty=100,
        epochs=60,
        use_xinference=False,  # 如果要走 HTTP，把它改成 True
        xinference_base_url="http://127.0.0.1:9997",
        pooling="mean",
    )
    pretrain_and_save_adapters(TRAIN_JSON, cfg)
