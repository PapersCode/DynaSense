# 机器标注结果

本目录保存 24,000 条机器标注结果，仅用于和人类标注结果进行比较，不是 gold label。

## 文件

| 文件 | 内容 |
|---|---|
| `machine_annotation_original.json` | 机器标注结果的 JSON 文件 |
| `machine_annotation_original.xlsx` | 按朝代整理的 Excel 文件 |

## 数据概览

| 类别 | 数值 |
|---|---:|
| 总条目数 | 24,000 |
| 来源文档数 | 3,536 |
| 朝代数 | 12 |
| 目标字数 | 244 |
| 非空机器标签数 | 23,616 |
| 空机器标签数 | 384 |

## JSON 格式

JSON 顶层包含三个字段：

| 字段 | 内容 |
|---|---|
| `metadata` | 文件来源和字段信息 |
| `rows` | 24,000 条机器标注记录 |
| `word_catalog` | 244 个目标字及候选义项 |

读取方式：

```python
import json

path = "data/machine_annotation_original/machine_annotation_original.json"

with open(path, "r", encoding="utf-8") as file:
    data = json.load(file)

rows = data["rows"]
word_catalog = data["word_catalog"]
```

`rows` 中每条记录包含：

```text
id, doc_id, text, word_id, word, options, label, dynasty, type
```

| 字段 | 类型 | 含义 |
|---|---|---|
| `id` | integer | 来源文档编号 |
| `doc_id` | string | 记录的唯一标识 |
| `text` | string | 古文上下文 |
| `word_id` | string | 目标字及其义项表编号 |
| `word` | string | 目标字 |
| `options` | string | 以分号连接的候选义项 |
| `label` | string 或 null | 源表 `correct_option_id` 中的机器标注结果 |
| `dynasty` | string | 原文所属朝代 |
| `type` | array | 统计分类 |

## Excel 格式

Excel 包含 `说明`、12 个朝代、`type汇总` 和 `word与备选义项`，共 15 个工作表。每个朝代工作表有 2,000 条记录。

## 与人类标注结果的区别

| 项目 | 机器标注结果 | 人类标注结果 |
|---|---|---|
| 路径 | `machine_annotation_original/` | `data/all_data.json` |
| `word_id` | 244 | 63 |
| 空标签 | 384 | 0 |
| `options` | 字符串 | 对象数组 |
| 训练/测试划分 | 无 | 19,200/4,800 |
