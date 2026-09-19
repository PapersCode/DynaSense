# DynaSense 数据说明

`data/` 目录包含当前发布版数据和历史机翻版数据。

## 文件

| 路径 | 内容 |
|---|---|
| `train.json` | 训练集，19,200 条 |
| `test.json` | 测试集，4,800 条 |
| `machine_translation_original/` | 历史机翻版及其说明 |
| `wsd_train.json` | 旧版 LoRA 占位文件，不作为当前训练数据 |

当前发布版使用 `train.json` 和 `test.json`。历史机翻版的数据结构不同，见 [`machine_translation_original/README.md`](machine_translation_original/README.md)。

## 数据概览

| 项目 | 数量 |
|---|---:|
| 记录 | 24,000 |
| `doc_id` | 24,000 |
| 来源文档 `id` | 3,524 |
| 朝代 | 12 |
| `word_id` | 63 |
| 目标字 | 63 |

## JSON 格式

`train.json` 和 `test.json` 的顶层均为数组。每条记录包含九个字段：

```text
id, doc_id, text, dynasty, word_id, word, options, label, type
```

示例：

```json
{
  "id": 104310,
  "doc_id": "doc_104310_95642285",
  "text": "……齐州保宁郡【兵】士屯于乐寿……",
  "dynasty": "宋",
  "word_id": "w10",
  "word": "兵",
  "options": [
    {"s1": "兵器"},
    {"s2": "军事，战争，或者有关军事的事情"},
    {"s3": "士兵，军队"}
  ],
  "label": "s3",
  "type": [
    {"C": "政治/制度语义驱动字"}
  ]
}
```

## 字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | integer | 来源文档编号，同一文档可对应多条记录 |
| `doc_id` | string | 记录的唯一标识 |
| `text` | string | 古文上下文，目标字用 `【】` 标记 |
| `dynasty` | string | 原文所属朝代 |
| `word_id` | string | 目标字及其义项表的编号 |
| `word` | string | 目标字 |
| `options` | array<object> | 候选义项，数组元素为 `{义项ID: 释义}` |
| `label` | string | 当前记录对应的义项 ID |
| `type` | array<object> | 统计分类，数组元素为 `{类型代码: 类型名称}` |

读取示例：

```python
import json

with open("data/train.json", "r", encoding="utf-8") as file:
    train = json.load(file)

text = train[0]["text"]
options = train[0]["options"]
label = train[0]["label"]
```

## 朝代与固定划分

| 朝代 | 训练集 | 测试集 | 合计 |
|---|---:|---:|---:|
| 西晋 | 1,600 | 400 | 2,000 |
| 北朝齐 | 1,600 | 400 | 2,000 |
| 元 | 1,600 | 400 | 2,000 |
| 后晋 | 1,600 | 400 | 2,000 |
| 东汉 | 1,600 | 400 | 2,000 |
| 南朝宋 | 1,600 | 400 | 2,000 |
| 南朝梁 | 1,600 | 400 | 2,000 |
| 西汉 | 1,600 | 400 | 2,000 |
| 清 | 1,600 | 400 | 2,000 |
| 明 | 1,600 | 400 | 2,000 |
| 宋 | 1,600 | 400 | 2,000 |
| 唐 | 1,600 | 400 | 2,000 |

## `type` 字段

`type` 是统计和分组字段，不是正确义项标签。当前共有六类：

| 代码 | 名称 | 记录数 |
|---|---|---:|
| A | 高频实词 | 9,582 |
| B | 语法化/抽象化字 | 5,173 |
| C | 政治/制度语义驱动字 | 3,262 |
| D | 社会身份/群体语义字 | 1,489 |
| E | 评价性/情态字 | 4,267 |
| F | 反直觉历史语义字 | 2,155 |

## 历史机翻版

历史机翻版位于 `machine_translation_original/`，包括 JSON 和分朝代 Excel。该版本保留早期机器标签，用于基线实验和版本比较。
