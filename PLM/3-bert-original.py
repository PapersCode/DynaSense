"""
词义消歧（WSD）系统 - 基于BERT的多分类头架构
支持每个word_id独立分类头，输出预测结果和评估报告
"""

import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    BertTokenizer, 
    BertModel, 
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import torch.serialization as serialization


@dataclass
class WSDConfig:
    """配置类"""
    model_name: str = "bert-base-chinese"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    train_ratio: float = 0.8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dropout: float = 0.1
    warmup_ratio: float = 0.1


class WSDDataset(Dataset):
    """WSD数据集类"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 提取文本和目标词
        text = str(item.get('text', ''))
        word = str(item.get('word', ''))
        word_id = str(item.get('word_id', ''))
        options = item.get('options', [])
        
        # 获取正确答案的索引
        correct_option_id = item.get('label')
        
        # 找到correct_option_id在options中的索引
        label_idx = -1
        if correct_option_id:
            correct_option_id = str(correct_option_id)
            for i, opt_dict in enumerate(options):
                if correct_option_id in opt_dict:
                    label_idx = i
                    break
        
        # 如果没找到标签，使用第一个选项（防止训练崩溃）
        if label_idx == -1 and len(options) > 0:
            label_idx = 0
        
        # 构造输入: [CLS] text [SEP] word [SEP]
        encoding = self.tokenizer(
            text,
            word,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'word_id': word_id,
            'label': label_idx,
            'num_options': len(options),
            'doc_id': item.get('doc_id', '')
        }


class MultiHeadWSDModel(nn.Module):
    """多分类头WSD模型 - 每个word_id一个独立分类头"""
    
    def __init__(self, bert_model_name: str, word_id_to_num_options: Dict[str, int], 
                 dropout: float = 0.1):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        
        # 为每个word_id创建独立的分类头
        self.classification_heads = nn.ModuleDict()
        for word_id, num_options in word_id_to_num_options.items():
            self.classification_heads[word_id] = nn.Linear(
                self.bert.config.hidden_size, 
                num_options
            )
    
    def forward(self, input_ids, attention_mask, word_ids, num_options_list):
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            word_ids: List[str] - 每个样本的word_id
            num_options_list: List[int] - 每个样本的选项数量
        """
        # BERT编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 使用[CLS]表示
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        cls_output = self.dropout(cls_output)
        
        # 路由到对应的分类头
        logits_list = []
        for i, (word_id, num_opts) in enumerate(zip(word_ids, num_options_list)):
            # 获取该样本对应的分类头
            head = self.classification_heads[word_id]
            logit = head(cls_output[i:i+1])  # [1, num_options]
            
            # 只取有效的选项数量
            logit = logit[:, :num_opts]
            logits_list.append(logit)
        
        return logits_list


class WSDTrainer:
    """WSD训练器"""
    
    def __init__(self, config: WSDConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # 初始化tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(config.model_name)
        
        # 数据统计
        self.word_id_to_num_options = {}
        self.word_id_to_options = {}
        
    def load_data(self, data_path: str) -> List[Dict]:
        """加载数据"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"[OK] 加载数据: {len(data)} 条")
        return data
    
    def prepare_data(self, data: List[Dict]) -> List[Dict]:
        #  Tuple[List[Dict], List[Dict]]
        """准备训练数据"""
        
        # 统计每个word_id的选项数量（取最大值）
        for item in data:
            word_id = item['word_id']
            num_options = len(item['options'])
            
            if word_id not in self.word_id_to_num_options:
                self.word_id_to_num_options[word_id] = num_options
                self.word_id_to_options[word_id] = item['options']
            else:
                self.word_id_to_num_options[word_id] = max(
                    self.word_id_to_num_options[word_id], 
                    num_options
                )
        
        print(f"✓ 发现 {len(self.word_id_to_num_options)} 个不同的word_id")
        
        # 过滤掉没有标签的数据
        valid_data = []
        for item in data:
            label_id = item.get('label')
            if label_id:
                valid_data.append(item)
        
        print(f"✓ 有效数据: {len(valid_data)} 条")
        
        # # 分割训练集和验证集
        # train_data, val_data = train_test_split(
        #     valid_data, 
        #     train_size=self.config.train_ratio, 
        #     random_state=42,
        #     shuffle=True
        # )
        
        print(f"✓ 训练集: {len(valid_data)} 条")
        # print(f"✓ 验证集: {len(val_data)} 条")
        
        return valid_data
    
    def create_dataloaders(self, train_data: List[Dict], val_data: List[Dict]) -> Tuple:
        """创建数据加载器"""
        
        train_dataset = WSDDataset(train_data, self.tokenizer, self.config.max_length)
        val_dataset = WSDDataset(val_data, self.tokenizer, self.config.max_length)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            collate_fn=self.collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            collate_fn=self.collate_fn
        )
        
        return train_loader, val_loader
    
    def collate_fn(self, batch):
        """自定义批处理函数"""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        word_ids = [item['word_id'] for item in batch]
        labels = torch.tensor([item['label'] for item in batch])
        num_options = [item['num_options'] for item in batch]
        doc_ids = [item['doc_id'] for item in batch]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'word_ids': word_ids,
            'labels': labels,
            'num_options': num_options,
            'doc_ids': doc_ids
        }
    
    def train(self, train_loader, val_loader, save_dir: str = "models"):
        """训练模型"""
        
        # 创建模型
        model = MultiHeadWSDModel(
            self.config.model_name,
            self.word_id_to_num_options,
            self.config.dropout
        ).to(self.device)
        
        # 优化器和学习率调度
        optimizer = AdamW(model.parameters(), lr=self.config.learning_rate)
        
        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # 训练循环
        best_val_acc = 0.0
        
        for epoch in range(self.config.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"{'='*60}")
            
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            progress_bar = tqdm(train_loader, desc="Training")
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                word_ids = batch['word_ids']
                labels = batch['labels'].to(self.device)
                num_options = batch['num_options']
                
                optimizer.zero_grad()
                
                # 前向传播
                logits_list = model(input_ids, attention_mask, word_ids, num_options)
                
                # 计算损失
                loss = 0
                for i, logits in enumerate(logits_list):
                    loss += nn.CrossEntropyLoss()(logits, labels[i:i+1])
                
                loss = loss / len(logits_list)
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # 统计
                train_loss += loss.item()
                
                for i, logits in enumerate(logits_list):
                    pred = torch.argmax(logits, dim=1)
                    train_correct += (pred == labels[i:i+1]).sum().item()
                    train_total += 1
                
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{train_correct/train_total:.4f}'
                })
            
            train_acc = train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # 验证阶段
            val_acc, val_loss = self.evaluate(model, val_loader)
            
            print(f"\nTrain Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'word_id_to_num_options': self.word_id_to_num_options,
                    'word_id_to_options': self.word_id_to_options,
                    'config': self.config
                }, f"{save_dir}/best_model.pt")
                print(f"✓ 保存最佳模型 (val_acc: {val_acc:.4f})")
        
        return model
    
    def evaluate(self, model, data_loader):
        """评估模型"""
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                word_ids = batch['word_ids']
                labels = batch['labels'].to(self.device)
                num_options = batch['num_options']
                
                logits_list = model(input_ids, attention_mask, word_ids, num_options)
                
                loss = 0
                for i, logits in enumerate(logits_list):
                    loss += nn.CrossEntropyLoss()(logits, labels[i:i+1])
                loss = loss / len(logits_list)
                
                val_loss += loss.item()
                
                for i, logits in enumerate(logits_list):
                    pred = torch.argmax(logits, dim=1)
                    val_correct += (pred == labels[i:i+1]).sum().item()
                    val_total += 1
        
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(data_loader)
        
        return val_acc, avg_val_loss


class WSDPredictor:
    """WSD预测器"""
    
    def __init__(self, model_path: str, config: WSDConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # 加载模型
        serialization.add_safe_globals([WSDConfig])
        checkpoint = torch.load(model_path, map_location=self.device)
        self.word_id_to_num_options = checkpoint['word_id_to_num_options']
        self.word_id_to_options = checkpoint['word_id_to_options']
        
        self.model = MultiHeadWSDModel(
            config.model_name,
            self.word_id_to_num_options,
            config.dropout
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.tokenizer = BertTokenizer.from_pretrained(config.model_name)
        
        print(f"✓ 模型加载成功: {model_path}")
    
    def predict(self, data: List[Dict], output_path: str):
        """预测并保存结果"""
        
        results = []
        
        dataset = WSDDataset(data, self.tokenizer, self.config.max_length)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            collate_fn=self._collate_fn
        )
        
        print("\n开始预测...")
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                word_ids = batch['word_ids']
                num_options = batch['num_options']
                doc_ids = batch['doc_ids']
                
                logits_list = self.model(input_ids, attention_mask, word_ids, num_options)
                
                for i, (logits, doc_id) in enumerate(zip(logits_list, doc_ids)):
                    pred_idx = torch.argmax(logits, dim=1).item()
                    
                    # 找到对应的原始数据
                    original_item = next(item for item in data if item['doc_id'] == doc_id)
                    
                    # 获取预测的option_id
                    options = original_item['options']
                    if pred_idx < len(options):
                        pred_option_id = list(options[pred_idx].keys())[0]
                    else:
                        pred_option_id = "unknown"
                    
                    # 复制原始数据并添加预测结果
                    result = original_item.copy()
                    result['model_option_id'] = pred_option_id
                    
                    results.append(result)
        
        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 预测结果已保存: {output_path}")
        
        return results
    
    def _collate_fn(self, batch):
        """自定义批处理"""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        word_ids = [item['word_id'] for item in batch]
        num_options = [item['num_options'] for item in batch]
        doc_ids = [item['doc_id'] for item in batch]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'word_ids': word_ids,
            'num_options': num_options,
            'doc_ids': doc_ids
        }


class WSDEvaluator:
    """WSD评估器 - 生成按朝代分类的报告"""
    
    @staticmethod
    def evaluate_and_report(predictions: List[Dict], output_xlsx: str):
        """评估并生成报告"""
        
        # 按朝代分组
        dynasty_groups = defaultdict(list)
        for item in predictions:
            dynasty = item.get('dynasty', 'unknown')
            dynasty_groups[dynasty].append(item)
        
        # 计算每个朝代的指标
        results = []
        
        for dynasty, items in dynasty_groups.items():
            y_true = []
            y_pred = []
            
            for item in items:
                # 真实标签
                true_id = item.get('label')
                pred_id = item.get('model_option_id')
                
                if true_id and pred_id:
                    y_true.append(true_id)
                    y_pred.append(pred_id)
            
            if len(y_true) > 0:
                accuracy = accuracy_score(y_true, y_pred)
                macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
                weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                
                results.append({
                    'dynasty': dynasty,
                    'accuracy': accuracy,
                    'macro_f1': macro_f1,
                    'weighted_f1': weighted_f1,
                    'num_samples': len(y_true)
                })
        
        # 转为DataFrame
        df = pd.DataFrame(results)
        
        # 计算总体均值
        mean_row = {
            'dynasty': 'MEAN',
            'accuracy': df['accuracy'].mean(),
            'macro_f1': df['macro_f1'].mean(),
            'weighted_f1': df['weighted_f1'].mean(),
            'num_samples': df['num_samples'].sum()
        }
        
        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
        
        # 保存为Excel
        df.to_excel(output_xlsx, index=False)
        
        print(f"\n{'='*60}")
        print("评估报告")
        print(f"{'='*60}")
        print(df.to_string(index=False))
        print(f"\n✓ 报告已保存: {output_xlsx}")
        
        return df


def main():
    """主函数"""
    
    # ========== 配置 ==========
    config = WSDConfig(
        model_name="/root/autodl-tmp/models/bert-base-chinese",
        max_length=512,
        batch_size=16,
        learning_rate=2e-5,
        num_epochs=10,
        train_ratio=0.8
    )
    
    # ========== 路径配置 ==========
    DATA_DIR = Path(__file__).resolve().parents[1] / "data"
    TRAIN_DATA_PATH = str(DATA_DIR / "train.json")
    TEST_DATA_PATH = str(DATA_DIR / "test.json")
    MODEL_SAVE_DIR = "/root/autodl-tmp/atd/data/结果/PLM/"
    PREDICTIONS_PATH = "/root/autodl-tmp/atd/data/结果/PLM/0406-bert-pred.json"
    REPORT_PATH = "/root/autodl-tmp/atd/data/结果/PLM/0406-bert-eval.xlsx"
    
    # ========== 训练模式 ==========
    TRAIN_MODE = True  # 设为False则只进行预测
    
    if TRAIN_MODE:
        print("\n" + "="*60)
        print("开始训练WSD模型")
        print("="*60)
        
        trainer = WSDTrainer(config)
        
        # 加载和准备数据
        train_data = trainer.load_data(TRAIN_DATA_PATH)
        val_data = trainer.load_data(TEST_DATA_PATH)
        train_data = trainer.prepare_data(train_data)
        val_data = trainer.prepare_data(val_data)
        train_loader, val_loader = trainer.create_dataloaders(train_data, val_data)
        
        # 训练
        model = trainer.train(train_loader, val_loader, MODEL_SAVE_DIR)
        
        print(f"\n✓ 训练完成! 模型保存在: {MODEL_SAVE_DIR}")
    
    # ========== 预测模式 ==========
    print("\n" + "="*60)
    print("开始预测")
    print("="*60)
    
    predictor = WSDPredictor(f"{MODEL_SAVE_DIR}/best_model.pt", config)
    
    # 加载测试数据（或全部数据）
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 预测
    predictions = predictor.predict(test_data, PREDICTIONS_PATH)
    
    # ========== 评估 ==========
    print("\n" + "="*60)
    print("生成评估报告")
    print("="*60)
    
    WSDEvaluator.evaluate_and_report(predictions, REPORT_PATH)
    
    print("\n✓ 全部完成!")


if __name__ == "__main__":
    main()
