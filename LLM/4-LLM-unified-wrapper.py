"""
4-LLM-unified-wrapper.py

功能: 将预训练的dynasty adapters与原始LLM集成，实现朝代感知的推理

使用流程:
    1. 加载LLM和adapter: wrapper = LLMWithAdapterWrapper(model_name, adapter_pt_path)
    2. 推理时传入dynasty: wrapper.generate(text, dynasty="唐")
    3. Adapter会自动融合到LLM的推理过程中

作者: AI Assistant
"""

import os
import json
import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLMWithAdapterWrapper:
    """
    将dynasty adapters动态注入到LLM中进行推理。
    
    核心特性：
    - 支持多种adapter注入方式（embedding融合、隐层hook等）
    - 自动处理dynasty词表匹配
    - 内置adapter缓存和fallback机制
    - 支持批量推理
    """
    
    def __init__(self, 
                 model_name: str,
                 adapter_pt_path: str,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 adapter_scale: float = 0.3):
        """
        初始化LLM+Adapter wrapper
        
        Args:
            model_name: HuggingFace模型名称或本地路径
            adapter_pt_path: dynasty_adapters.pt文件路径
            device: 计算设备 ("cuda" 或 "cpu")
            adapter_scale: adapter对最终表示的影响权重 (0.0-1.0)
        """
        self.device = torch.device(device)
        self.model_name = model_name
        self.adapter_scale = adapter_scale
        
        # 1. 加载原始LLM和tokenizer
        print(f"[AdapterWrapper] 加载LLM: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.model.eval()
        self.hidden_size = self.model.config.hidden_size
        
        # 2. 加载adapter数据
        print(f"[AdapterWrapper] 加载adapter: {adapter_pt_path}")
        if not os.path.exists(adapter_pt_path):
            raise FileNotFoundError(f"Adapter文件不存在: {adapter_pt_path}")
        
        checkpoint = torch.load(adapter_pt_path, map_location="cpu")
        
        # 解析adapter数据结构
        self.dynasty_to_idx = checkpoint.get("dynasty_to_idx", {})
        self.label_to_idx = checkpoint.get("label_to_idx", {})
        self.dynasty_adapters = checkpoint.get("dynasty_adapters", {})
        self.adapter_config = checkpoint.get("hyper_config", {})
        
        # 将adapter向量移到GPU
        for dynasty, adapter_vec in self.dynasty_adapters.items():
            if isinstance(adapter_vec, torch.Tensor):
                self.dynasty_adapters[dynasty] = adapter_vec.to(self.device)
        
        print(f"[AdapterWrapper] 已加载 {len(self.dynasty_adapters)} 个朝代adapter")
        
        # 获取adapter维度信息
        if self.dynasty_adapters:
            first_adapter = next(iter(self.dynasty_adapters.values()))
            self.adapter_dim = first_adapter.shape[-1]
            print(f"[AdapterWrapper] adapter维度: {self.adapter_dim}")
        else:
            print("[警告] 未找到任何adapter")
            self.adapter_dim = None
        
        # 3. 初始化adapter融合层
        if self.adapter_dim is not None and self.adapter_dim == self.hidden_size:
            self.fusion_layer = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size)
            ).to(self.device)
        elif self.adapter_dim is not None:
            # 如果维度不匹配，创建投影层
            print(f"[信息] adapter_dim ({self.adapter_dim}) != hidden_size ({self.hidden_size}), 将创建投影层")
            self.adapter_proj = nn.Linear(self.adapter_dim, self.hidden_size).to(self.device)
            self.fusion_layer = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size)
            ).to(self.device)
        else:
            self.fusion_layer = None
        
        # 4. Adapter缓存（避免重复lookup）
        self._adapter_cache = {}
        
        print(f"[AdapterWrapper] 初始化完成 (device={device})")
    
    def get_available_dynasties(self) -> List[str]:
        """获取所有可用的朝代列表"""
        return list(self.dynasty_adapters.keys())
    
    def get_adapter(self, dynasty: str) -> Optional[torch.Tensor]:
        """
        获取指定朝代的adapter向量
        
        Args:
            dynasty: 朝代名称
            
        Returns:
            adapter向量 [adapter_dim] 或 None
        """
        # 检查缓存
        if dynasty in self._adapter_cache:
            return self._adapter_cache[dynasty]
        
        # 精确匹配
        adapter = self.dynasty_adapters.get(str(dynasty))
        
        if adapter is not None:
            self._adapter_cache[dynasty] = adapter
            return adapter
        
        # 如果未找到，返回None（调用端会处理fallback）
        return None
    
    def _get_average_adapter(self) -> torch.Tensor:
        """获取所有adapter的平均值（用作fallback）"""
        if not self.dynasty_adapters:
            return None
        
        adapters = list(self.dynasty_adapters.values())
        return torch.stack(adapters).mean(0)
    
    def forward_with_adapter(self, 
                            input_ids: torch.Tensor,
                            attention_mask: Optional[torch.Tensor] = None,
                            dynasty: Optional[str] = None,
                            use_cache: bool = False,
                            **kwargs):
        """
        带adapter的前向传播
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            dynasty: 朝代标签
            use_cache: 是否使用KV缓存
            
        Returns:
            模型输出
        """
        # 获取原始LLM输出
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=use_cache,
            **kwargs
        )
        
        # 如果指定了dynasty，应用adapter
        if dynasty is not None and self.fusion_layer is not None:
            adapter = self.get_adapter(dynasty)
            
            if adapter is None:
                print(f"[警告] dynasty '{dynasty}' 未找到，使用平均adapter")
                adapter = self._get_average_adapter()
            
            if adapter is not None:
                # 获取最后一层的隐层表示
                hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
                batch_size, seq_len, _ = hidden_states.shape
                
                # 投影adapter（如果需要）
                if hasattr(self, 'adapter_proj'):
                    adapter = self.adapter_proj(adapter)
                
                # 广播adapter到整个batch和序列
                adapter_expanded = adapter.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size]
                adapter_expanded = adapter_expanded.expand(batch_size, seq_len, -1)  # [B, T, H]
                
                # 融合: h' = h + scale * fusion(h + adapter)
                fused_hidden = hidden_states + self.adapter_scale * adapter_expanded
                fused_hidden = self.fusion_layer(fused_hidden)
                
                # 更新hidden_states
                outputs.hidden_states = outputs.hidden_states[:-1] + (fused_hidden,)
        
        return outputs
    
    def generate(self, 
                text: str, 
                dynasty: Optional[str] = None,
                max_length: int = 256,
                temperature: float = 0.7,
                top_p: float = 0.9,
                do_sample: bool = True,
                **kwargs) -> str:
        """
        带adapter的文本生成
        
        Args:
            text: 输入文本
            dynasty: 朝代标签（用于选择adapter）
            max_length: 最大生成长度
            temperature: 温度参数
            top_p: nucleus采样参数
            do_sample: 是否采样（False为贪心解码）
            
        Returns:
            生成的文本
        """
        # Tokenize输入
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask')
        
        # 方案A: 通过修改input_embeds来融合adapter
        if dynasty is not None and self.fusion_layer is not None:
            adapter = self.get_adapter(dynasty)
            
            if adapter is None:
                adapter = self._get_average_adapter()
            
            if adapter is not None:
                # 获取输入embedding
                input_embeds = self.model.get_input_embeddings()(input_ids)
                
                # 投影adapter
                if hasattr(self, 'adapter_proj'):
                    adapter = self.adapter_proj(adapter)
                
                # 融合: emb' = emb + scale * adapter
                adapter_repeated = adapter.unsqueeze(0).unsqueeze(0).expand(
                    input_embeds.shape[0], input_embeds.shape[1], -1
                )
                input_embeds = input_embeds + self.adapter_scale * adapter_repeated
                
                # 使用修改后的embedding生成
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs_embeds=input_embeds,
                        attention_mask=attention_mask,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        **kwargs
                    )
            else:
                # 无adapter的标准生成
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        **kwargs
                    )
        else:
            # 无adapter的标准生成
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
        
        # 解码结果
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result
    
    def batch_generate(self,
                      texts: List[str],
                      dynasty: Optional[str] = None,
                      max_length: int = 256,
                      **kwargs) -> List[str]:
        """
        批量生成（针对相同dynasty的优化）
        
        Args:
            texts: 输入文本列表
            dynasty: 朝代标签（对所有文本应用同一dynasty）
            max_length: 最大生成长度
            
        Returns:
            生成文本列表
        """
        results = []
        for text in texts:
            result = self.generate(
                text=text,
                dynasty=dynasty,
                max_length=max_length,
                **kwargs
            )
            results.append(result)
        return results
    
    def get_adapter_info(self) -> Dict:
        """获取adapter的统计信息"""
        return {
            "num_dynasties": len(self.dynasty_adapters),
            "dynasties": list(self.dynasty_adapters.keys()),
            "adapter_dim": self.adapter_dim,
            "hidden_size": self.hidden_size,
            "adapter_scale": self.adapter_scale,
            "model_name": self.model_name,
        }


# ========================================
# 测试代码
# ========================================

def test_wrapper():
    """测试wrapper的基本功能"""
    
    # 配置
    MODEL_NAME = "/root/autodl-tmp/models/Qwen3-1.5B-Instruct"
    ADAPTER_PT = "/root/autodl-tmp/atd/data/结果/LLM/dynasty_adapters.pt"
    
    try:
        # 1. 初始化wrapper
        print("\n=== 初始化wrapper ===")
        wrapper = LLMWithAdapterWrapper(
            model_name=MODEL_NAME,
            adapter_pt_path=ADAPTER_PT,
            device="cuda" if torch.cuda.is_available() else "cpu",
            adapter_scale=0.3
        )
        
        # 2. 查看可用朝代
        print("\n=== 可用朝代 ===")
        dynasties = wrapper.get_available_dynasties()
        print(f"朝代列表: {dynasties}")
        
        # 3. 获取adapter信息
        print("\n=== Adapter信息 ===")
        info = wrapper.get_adapter_info()
        for k, v in info.items():
            print(f"{k}: {v}")
        
        # 4. 测试生成（带adapter）
        if dynasties:
            print("\n=== 测试生成（带adapter） ===")
            test_text = "古文: 子曰"
            dynasty = dynasties[0]
            print(f"输入: {test_text}")
            print(f"朝代: {dynasty}")
            
            result = wrapper.generate(
                text=test_text,
                dynasty=dynasty,
                max_length=128,
                temperature=0.5,
                do_sample=False
            )
            print(f"输出: {result}")
        
        # 5. 测试生成（不带adapter）
        print("\n=== 测试生成（不带adapter） ===")
        test_text = "古文: 子曰"
        result = wrapper.generate(
            text=test_text,
            dynasty=None,
            max_length=128,
            temperature=0.5,
            do_sample=False
        )
        print(f"输出: {result}")
        
        print("\n✓ 测试完成")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_wrapper()
