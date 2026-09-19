# 历史机翻版

本目录保存 DynaSense 的历史机翻数据，用于基线实验和版本比较。当前发布数据位于上一级目录的 `train.json` 和 `test.json`。

## 文件

| 文件 | 内容 |
|---|---|
| `machine_translation_original.json` | 程序读取用 JSON |
| `machine_translation_original.xlsx` | 按朝代整理的 Excel |

## 数据概览

| 项目 | 数量 |
|---|---:|
| 记录 | 24,000 |
| `doc_id` | 24,000 |
| 来源文档 `id` | 3,536 |
| 朝代 | 12 |
| `word_id` | 244 |
| 目标字 | 244 |
| 非空机器标签 | 23,616 |
| 空机器标签 | 384 |

## JSON 格式

JSON 顶层包含三个字段：

| 字段 | 内容 |
|---|---|
| `metadata` | 文件来源和字段信息 |
| `rows` | 24,000 条数据记录 |
| `word_catalog` | 244 个目标字及候选义项 |

读取示例：

```python
import json

path = "data/machine_translation_original/machine_translation_original.json"

with open(path, "r", encoding="utf-8") as file:
    data = json.load(file)

rows = data["rows"]
word_catalog = data["word_catalog"]
```

`rows` 中每条记录包含：

```text
id, doc_id, text, word_id, word, options, label, dynasty, type
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | integer | 来源文档编号 |
| `doc_id` | string | 记录的唯一标识 |
| `text` | string | 古文上下文 |
| `word_id` | string | 目标字及其义项表编号 |
| `word` | string | 目标字 |
| `options` | string | 以分号连接的候选义项 |
| `label` | string 或 null | 机器选择的义项 ID |
| `dynasty` | string | 原文所属朝代 |
| `type` | array<object> | 统计分类；未分类记录使用 U |

## Excel 格式

Excel 包含 15 个工作表：

- `说明`
- 12 个朝代工作表
- `type汇总`
- `word与备选义项`

每个朝代工作表有 2,000 条记录。

## 与当前发布版的区别

| 项目 | 历史机翻版 | 当前发布版 |
|---|---|---|
| 路径 | `machine_translation_original/` | `data/train.json`、`data/test.json` |
| 标签 | 机器标签 | 当前发布标签 |
| `word_id` | 244 | 63 |
| 空标签 | 384 | 0 |
| `options` | 字符串 | 对象数组 |
| 训练/测试划分 | 无 | 19,200/4,800 |
