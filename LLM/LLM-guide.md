## 📋 概览

| 文件 | 说明 |
|------|------|
| `4-LLM-unified-wrapper.py` | **Adapter+LLM集成层**（核心模块） |
| `4-LLM-main-integrated.py` | **改造后主任务**（调用wrapper） |
| `4-LLM-adapter.py` | 现有的adapter生成脚本（保持不变） |

---

## 🚀 快速开始

### 步骤1: 准备文件

```bash
# 将以下文件放到你的项目目录
cp 4-LLM-unified-wrapper.py /root/autodl-tmp/atd/代码/
cp 4-LLM-main-integrated.py /root/autodl-tmp/atd/代码/
```

### 步骤2: 验证依赖

确保已安装必要的库：
```bash
pip install torch transformers tqdm
```

### 步骤3: 配置路径

编辑 `4-LLM-main-integrated.py`，确保以下路径正确：

```python
# 第40-42行
MODEL_NAME = "/root/autodl-tmp/models/Qwen3-1.5B-Instruct"  # LLM模型路径
ADAPTER_PT_PATH = "/root/autodl-tmp/atd/data/结果/LLM/dynasty_adapters.pt"  # adapter文件
JSON_PATH = "/root/autodl-tmp/atd/data/人工标注/最终结果/merged_dataset_Type_flattened.json"  # 输入数据
```

### 步骤4: 验证数据格式

确保输入JSON包含 `dynasty` 字段：

```json
{
  "index": 0,
  "label": "唐",  // ← 必需
  "text": "古文原文...",
  "word_matches": [...]
}
```

### 步骤5: 运行

```bash
cd /root/autodl-tmp/atd/代码/
python 4-LLM-main-integrated.py
```

---

## 📊 流程图

```
输入JSON (包含dynasty)
    ↓
process_item() 提取dynasty
    ↓
translate_single_word(dynasty=...)
    ↓
call_model(..., dynasty=...)
    ↓
llm_wrapper.generate(text, dynasty)
    ↓
LLMWithAdapterWrapper:
  ├─ 加载adapter by dynasty ✓
  ├─ 融合到embedding/hidden states
  └─ 返回推理结果
    ↓
输出JSON (含correct_option)
```

---

## 🔧 核心改动详解

### 1️⃣ 初始化时加载adapter

**原来**：
```python
# 使用xinference
client = RESTfulClient(XINFERENCE_URL)
model = client.get_model(MODEL_UID)
```

**现在**：
```python
# 使用adapter wrapper
llm_wrapper = LLMWithAdapterWrapper(
    model_name=MODEL_NAME,
    adapter_pt_path=ADAPTER_PT_PATH,
    adapter_scale=0.3  # 可调整adapter影响权重
)
use_adapter = True
```

### 2️⃣ 函数签名添加dynasty参数

**原来**：
```python
def translate_single_word(idx, text, word, translation, options_list, max_retries=3):
    ...
```

**现在**：
```python
def translate_single_word(idx, text, word, translation, options_list, 
                         dynasty=None, max_retries=3):  # 新增dynasty参数
    ...
```

### 3️⃣ 调用时传递dynasty

**原来**：
```python
future = inner_executor.submit(translate_single_word, idx, text, word, translation, options_list)
```

**现在**：
```python
# 从item中提取dynasty
dynasty = item.get("dynasty") or item.get("period") or None

# 传入调用
future = inner_executor.submit(
    translate_single_word, 
    idx, text, word, translation, options_list,
    dynasty=dynasty  # 传入朝代
)
```

### 4️⃣ 模型调用集成adapter

**原来**：
```python
def call_model(messages, timeout=MODEL_CALL_TIMEOUT):
    completion = model.chat(messages, generate_config={...})
    return completion["choices"][0]["message"]["content"]
```

**现在**：
```python
def call_model(messages, timeout=MODEL_CALL_TIMEOUT, dynasty=None):
    if use_adapter and llm_wrapper is not None:
        reply = llm_wrapper.generate(
            text=full_prompt,
            dynasty=dynasty,  # 传入朝代标签
            max_length=2048,
            temperature=0.5
        )
        return reply
    else:
        # fallback到xinference
        ...
```

---

## ⚙️ 配置参数

### LLMWithAdapterWrapper 参数

```python
wrapper = LLMWithAdapterWrapper(
    model_name="/path/to/model",          # HF模型名或本地路径
    adapter_pt_path="/path/to/adapters.pt",  # adapter文件路径
    device="cuda",                         # "cuda" 或 "cpu"
    adapter_scale=0.3                      # adapter影响权重 (0.0-1.0)
                                          # 越大adapter影响越强
)
```

### 生成参数

```python
wrapper.generate(
    text="输入文本",
    dynasty="唐",                  # 朝代标签
    max_length=256,               # 最大生成长度
    temperature=0.5,              # 0.0=贪心, 1.0=随机
    top_p=0.9,                   # nucleus采样
    do_sample=True               # 是否采样
)
```

---

## 🐛 常见问题

### Q1: "adapter文件不存在"

**原因**：adapter_pt_path 路径错误

**解决**：
```bash
# 检查文件是否存在
ls -lh /root/autodl-tmp/atd/data/结果/LLM/dynasty_adapters.pt

# 确认adapter已生成
python 4-LLM-adapter.py
```

### Q2: "dynasty 'xxx' 未找到"

**原因**：输入数据中的dynasty与adapter vocab不一致

**解决**：
```python
# 查看可用朝代
wrapper = LLMWithAdapterWrapper(...)
print(wrapper.get_available_dynasties())

# 检查输入数据中的dynasty值
with open(JSON_PATH) as f:
    data = json.load(f)
    dynasties_in_data = set(d.get("dynasty") for d in data)
    print("数据中的朝代:", dynasties_in_data)
```

### Q3: 内存溢出 (OOM)

**原因**：显存不足

**解决**：
```python
# 方案A: 减小batch_size
batch_size = 25  # 改小

# 方案B: 启用梯度检查点（降低显存）
llm_wrapper.model.gradient_checkpointing_enable()

# 方案C: 使用CPU推理
llm_wrapper = LLMWithAdapterWrapper(..., device="cpu")
```

### Q4: adapter维度不匹配

**原因**：adapter_dim != model.hidden_size

**解决**：自动处理（wrapper中已添加投影层）

```python
# wrapper会自动检测并添加投影层
if adapter_dim != hidden_size:
    self.adapter_proj = nn.Linear(adapter_dim, hidden_size)
```

### Q5: 性能慢

**原因**：adapter融合计算开销

**解决**：
```python
# 方案A: 减小adapter_scale（更快但效果弱）
wrapper = LLMWithAdapterWrapper(..., adapter_scale=0.1)

# 方案B: 启用缓存
# wrapper已内置adapter缓存，自动优化

# 方案C: 批量处理相同dynasty
results = wrapper.batch_generate(texts, dynasty="唐")
```

---

## ✅ 测试checklist

运行前验证：

```python
import torch
from unified_adapter import LLMWithAdapterWrapper

# 1. 检查adapter文件
import os
assert os.path.exists(ADAPTER_PT_PATH), "adapter文件不存在"
print("✓ Adapter文件存在")

# 2. 初始化wrapper
wrapper = LLMWithAdapterWrapper(MODEL_NAME, ADAPTER_PT_PATH)
print("✓ Wrapper初始化成功")

# 3. 查看adapter信息
info = wrapper.get_adapter_info()
print(f"✓ Adapter信息: {info}")

# 4. 测试生成（带adapter）
dynasties = wrapper.get_available_dynasties()
if dynasties:
    result = wrapper.generate("测试文本", dynasty=dynasties[0], max_length=50)
    print(f"✓ 生成测试通过: {result[:50]}")

# 5. 检查数据格式
import json
with open(JSON_PATH) as f:
    data = json.load(f)
    sample = data[0]
    assert "dynasty" in sample, "数据缺少dynasty字段"
    assert "word_matches" in sample, "数据缺少word_matches字段"
    print(f"✓ 数据格式正确 (sample dynasty: {sample.get('dynasty')})")

print("\n✓ 所有检查通过！可以开始运行")
```

---

## 📈 性能优化

### 方案A: Batch推理（推荐）

```python
# 对同一朝代的多条文本合并处理
texts = ["文本1", "文本2", "文本3"]
results = wrapper.batch_generate(texts, dynasty="唐")
```

### 方案B: Adapter缓存

```python
# wrapper自动缓存已加载的adapter
adapter1 = wrapper.get_adapter("唐")  # 第一次加载
adapter2 = wrapper.get_adapter("唐")  # 第二次从缓存取（速度快）
```

### 方案C: 调整adapter_scale

```python
# adapter_scale越小越快，但效果可能弱
wrapper = LLMWithAdapterWrapper(..., adapter_scale=0.1)  # 快速模式
wrapper = LLMWithAdapterWrapper(..., adapter_scale=0.5)  # 平衡
wrapper = LLMWithAdapterWrapper(..., adapter_scale=1.0)  # 最强效果
```

---

## 📝 输出格式

输出JSON结构：

```json
{
  "index": 0,
  "dynasty": "唐",
  "text": "古文...",
  "word_matches": [
    {
      "id": "wm001",
      "word": "字",
      "translation": "翻译",
      "options": [...],
      "correct_option": "s1"  // ← adapter推理的结果
    }
  ]
}
```

---

## 🔍 调试技巧

### 启用详细日志

```python
# 修改logging level
logger.setLevel(logging.DEBUG)

# 或在初始化时添加
logging.basicConfig(level=logging.DEBUG)
```

### 查看adapter应用情况

```python
# 在process_item中添加
logger.info(f"dynasty={dynasty}, 适用adapter={llm_wrapper.get_adapter(dynasty) is not None}")
```

### 单条样本测试

```python
# 测试单个word_match
idx = 0
text = "古文..."
word = "字"
translation = "翻译"
options = [{"s1": "义项1"}]
dynasty = "唐"

result = translate_single_word(idx, text, word, translation, options, dynasty)
print(f"结果: {result}")
```

---

## 📊 监控指标

关键日志字段：

```
[LLMWithAdapterWrapper]
  ├─ model_name: 使用的模型
  ├─ adapter_dim: adapter维度
  ├─ num_dynasties: dynasty数量
  └─ adapter_scale: 影响权重

[call_model]
  ├─ dynasty: 使用的朝代
  ├─ sense_id: 返回的义项ID
  └─ confidence: 置信度

[process_item]
  ├─ index: 记录索引
  ├─ dynasty: 本条朝代
  └─ word_matches: 处理数量
```
