# 历史机翻版数据说明

本目录保存 DynaSense 的历史机翻版本，供基线复现、误差分析和版本对照使用。

> **重要：本目录不是当前发布数据。** 需要训练或评估当前正式模型时，应使用上一级目录中的 `train.json` 和 `test.json`。不要把本目录中的机器标签与当前发布标签混合使用。

## 文件一览

| 文件 | 用途 | 是否建议直接训练 |
|---|---|---|
| `machine_translation_original.json` | 机器原始标签的无损 JSON 快照；程序处理时以此文件为准 | 仅适合作为机器基线，需先处理空标签和不同的候选项格式 |
| `machine_translation_original.xlsx` | 同一批数据的分朝代可读版本，便于人工查看和核对 | 否；程序训练优先读取 JSON |
| `README.md` | 本版本的结构、限制、读取和对比规则 | — |

文件完整性校验值：

| 文件 | SHA-256 |
|---|---|
| `machine_translation_original.json` | `99d5ec29e02cb41bdff7c0b153ee96df9afd10bae758e44ab6dd9757fc1ebed0` |
| `machine_translation_original.xlsx` | `aab0f4e03bd237a68887179976adeded029afef3d60eb8e0e9e0d3c26a3b5f70` |

## 版本定位

此版本来自早期 `word_selection(6).xlsx` 中的机器处理结果：

- `label` 原样取自源表的 `correct_option_id`；
- 没有使用当前发布版标签补写或替换；
- 23,616 条记录具有机器标签；
- 384 条记录的机器标签原本为空，JSON 中继续保存为 `null`，Excel 中继续留空；
- 非空机器标签均能在该记录的候选义项中找到；
- 文本里的字面转义 `\【`、`\】` 已规范为真实的 `【`、`】`；
- `word` 中仅用于显示的反斜杠和方头括号已清理；
- `type` 是为了统计和对照后加的辅助分类，不是原机器模型的输出。
- `metadata.source_xlsx` 只保留原始文件名，不包含生成环境中的本地绝对路径。

这里的“原始”表示机器标签没有被当前发布版覆盖，不表示文件字节与最早 Excel 完全相同。JSON 和本目录 Excel 是在不改机器标签的前提下生成的规范化快照。

## 数据规模

| 指标 | 数值 |
|---|---:|
| 总记录数 | 24,000 |
| 唯一 `doc_id` 数 | 24,000 |
| 唯一来源 `id` 数 | 3,536 |
| 朝代数 | 12 |
| 唯一 `word_id` 数 | 244 |
| 唯一目标字数 | 244 |
| 非空机器标签数 | 23,616 |
| 空机器标签数 | 384 |
| 已获得 A–F 分类的记录数 | 23,325 |
| `U` 未分类记录数 | 675 |
| 同时具有多个分类的记录数 | 2,302 |

12 个朝代各有 2,000 条记录：西晋、北朝齐、元、后晋、东汉、南朝宋、南朝梁、西汉、清、明、宋、唐。

本版本没有官方的训练集/测试集划分。不要自行切分后把结果称为仓库的正式划分；仓库正式划分只对应上一级目录的当前发布版。

## JSON 顶层结构

`machine_translation_original.json` 的顶层是对象，不是记录数组：

```text
metadata
rows
word_catalog
```

含义如下：

| 顶层键 | JSON 类型 | 含义 |
|---|---|---|
| `metadata` | object | 来源、标签来源、字段顺序、朝代顺序和未分类标记等生成信息 |
| `rows` | array<object> | 24,000 条机翻版 WSD 记录 |
| `word_catalog` | array<object> | 244 个 `word_id` 对应的目标字和候选义项目录 |

因此不能把顶层对象直接传给只接受样本数组的数据加载器。正确入口是：

```python
import json
from pathlib import Path

path = Path("data/machine_translation_original/machine_translation_original.json")

with path.open("r", encoding="utf-8") as handle:
    payload = json.load(handle)

rows = payload["rows"]
word_catalog = payload["word_catalog"]

assert len(rows) == 24_000
assert len(word_catalog) == 244
```

## `rows` 记录结构

每条记录按以下顺序保存九个字段：

```text
id, doc_id, text, word_id, word, options, label, dynasty, type
```

注意：这里的字段顺序以及 `options`、`label` 的类型与当前发布版不完全相同。

| 字段 | JSON 类型 | 可空 | 含义与处理要求 |
|---|---|---|---|
| `id` | integer | 否 | 来源古文文档编号。它会重复，不能作为样本主键。 |
| `doc_id` | string | 否 | 当前记录的唯一标识，格式为 `doc_<id>_<8位数字>`。合并和对照必须使用此字段。 |
| `text` | string | 否 | 古文上下文；目标字至少一次以 `【word】` 形式标出。 |
| `word_id` | string | 否 | 机器版义项表中的目标字标识，共 244 个。 |
| `word` | string | 否 | 需要消歧的目标字。 |
| `options` | string | 否 | 以全角分号连接的“义项 ID: 释义”字符串，不是 JSON 数组。 |
| `label` | string 或 null | 是 | 机器选择的义项 ID；384 条原始空值保存为 `null`。它不是当前发布版的参考标签。 |
| `dynasty` | string | 否 | 原文所属朝代。 |
| `type` | array<object> | 否 | 后加的统计分类；可包含一个或多个 A–F 类别，无法映射时为 U。 |

### `doc_id` 是唯一连接键

`id`、数组下标和 Excel 行号都不能代替 `doc_id`：

- `id` 在 24,000 条记录中只有 3,536 个不同值；
- 数组下标只反映当前文件顺序；
- 机翻版与当前发布版各有一部分独有记录，不能按位置逐行比较；
- 只有 `doc_id` 适合进行交集、差集和预测结果合并。

## `options` 的解析

机翻版的 `options` 是字符串，例如：

```text
s1: 义项一；s2: 义项二；s3: 义项三
```

义项 ID 并不保证只有 `s1`、`s2` 这一种形式。解析时应识别每个分隔符后、冒号前的实际 ID，并保留原顺序。下面的函数会转成与当前发布版相近的数组结构：

```python
import re

OPTION_START = re.compile(
    r"(?:^|[;；]\s*)([A-Za-z0-9]+(?:-\d+)?)\s*:\s*"
)


def parse_machine_options(value: str) -> list[dict[str, str]]:
    matches = list(OPTION_START.finditer(value))
    if not matches:
        raise ValueError(f"cannot parse options: {value!r}")

    parsed = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(value)
        definition = value[start:end].strip(" ;；")
        if not definition:
            raise ValueError(f"empty definition for {match.group(1)}")
        parsed.append({match.group(1): definition})
    return parsed
```

不要直接用 `value.split("；")` 后再假设义项 ID 只有一种格式，也不要把结果转为普通字典后去重。保留数组顺序更适合与现有分类代码对接。

### 把机器标签转换为分类索引

```python
def machine_label_to_index(row: dict) -> int | None:
    if row["label"] is None:
        return None

    options = parse_machine_options(row["options"])
    for index, entry in enumerate(options):
        if row["label"] in entry:
            return index
    raise ValueError(f"label outside options: {row['doc_id']}")
```

用于监督训练时，应显式选择如何处理 `None`。最安全的默认做法是跳过，不要把空标签改成第一个义项、空字符串或新的类别：

```python
labeled_rows = [row for row in rows if row["label"] is not None]
assert len(labeled_rows) == 23_616
```

## `type` 字段

`type` 用于统计，不是机器标签的一部分。其结构为对象数组：

```json
[
  {"A": "高频实词"},
  {"B": "语法化/抽象化字"}
]
```

类别含义：

| 代码 | 名称 | 说明 |
|---|---|---|
| A | 高频实词 | 高频、常见且具有多个实词义项的字 |
| B | 语法化/抽象化字 | 语法化、功能化或抽象化程度较高的字 |
| C | 政治/制度语义驱动字 | 义项与政治、军事或制度语境关系较强的字 |
| D | 社会身份/群体语义字 | 义项与身份、宗族或群体关系较强的字 |
| E | 评价性/情态字 | 带评价、判断、意愿或情态色彩的字 |
| F | 反直觉历史语义字 | 历史义与现代常用义差异较明显的字 |
| U | 未分类（机器原始版本无 type） | 机器版中无法从现有分类表映射的目标字 |

一条记录可能同时属于多个 A–F 类别。统计时应遍历整个数组，而不是只取第一个元素。`U` 只表示缺少现有分类映射，不能推断为新的语言学类别。

## Excel 工作簿结构

`machine_translation_original.xlsx` 有 15 个工作表，顺序为：

```text
说明
西晋、北朝齐、元、后晋、东汉、南朝宋、南朝梁、西汉、清、明、宋、唐
type汇总
word与备选义项
```

- 12 个朝代工作表各有 2,000 条数据；
- 各朝代表使用九列：`id`、`doc_id`、`text`、`word_id`、`word`、`options`、`label`、`dynasty`、`type`；
- `type汇总` 汇总 A–F 与 U 的目标字、`word_id`、总记录数和分朝代数量；
- `word与备选义项` 完整列出 244 个 `word_id`、目标字及候选义项；
- Excel 与 JSON 均包含同一批 24,000 条记录；
- Excel 主要供浏览和人工检查，程序处理应优先使用 JSON，避免单元格显示、换行和类型自动转换带来的歧义。

## 与当前发布版的区别

| 项目 | 历史机翻版 | 当前发布版 |
|---|---|---|
| 路径 | `data/machine_translation_original/` | `data/train.json`、`data/test.json` |
| 标签来源 | 机器生成的 `correct_option_id` | 当前发布的 `label` |
| 总记录数 | 24,000 | 24,000 |
| 唯一 `word_id` | 244 | 63 |
| 空标签 | 384 | 0 |
| `options` 类型 | 分号连接的字符串 | 有序的单键对象数组 |
| 顶层 JSON | 含 `metadata`、`rows`、`word_catalog` 的对象 | 记录数组 |
| 官方训练/测试划分 | 无 | 19,200/4,800 固定划分 |
| 推荐用途 | 历史基线、误差分析 | 正式训练和评估 |

两版不能按数组下标直接比较。按 `doc_id` 对齐后的统计如下：

| 对齐指标 | 数量 |
|---|---:|
| 两版共有的 `doc_id` | 20,277 |
| 仅机翻版存在的 `doc_id` | 3,723 |
| 仅当前发布版存在的 `doc_id` | 3,723 |
| 共有记录中机器非空标签与当前标签相同 | 11,279 |
| 共有记录中机器非空标签与当前标签不同 | 8,698 |
| 共有记录中机器标签为空、当前标签非空 | 300 |

这些数字不能直接视为版本修改数量。两版各有 3,723 条独有记录，候选义项结构也不完全相同；这里仅报告 `doc_id` 交集上的标签字符串对照。

正确的对照方式：

```python
import json
from pathlib import Path


def load_json(path: str):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


machine_payload = load_json(
    "data/machine_translation_original/machine_translation_original.json"
)
machine_by_doc_id = {
    row["doc_id"]: row for row in machine_payload["rows"]
}

current_rows = load_json("data/train.json") + load_json("data/test.json")
current_by_doc_id = {row["doc_id"]: row for row in current_rows}

shared_doc_ids = machine_by_doc_id.keys() & current_by_doc_id.keys()
machine_only = machine_by_doc_id.keys() - current_by_doc_id.keys()
current_only = current_by_doc_id.keys() - machine_by_doc_id.keys()

assert len(shared_doc_ids) == 20_277
assert len(machine_only) == 3_723
assert len(current_only) == 3_723
```

## 最小严格校验

下面的检查适合放在预处理入口，防止程序把机翻版误当成当前发布版或错误解析空标签：

```python
import json
import re
from pathlib import Path

DOC_ID_RE = re.compile(r"^doc_\d+_\d{8}$")
EXPECTED_FIELDS = [
    "id", "doc_id", "text", "word_id", "word",
    "options", "label", "dynasty", "type",
]

path = Path("data/machine_translation_original/machine_translation_original.json")
with path.open("r", encoding="utf-8") as handle:
    payload = json.load(handle)

assert list(payload) == ["metadata", "rows", "word_catalog"]
rows = payload["rows"]
assert len(rows) == 24_000
assert len({row["doc_id"] for row in rows}) == 24_000
assert sum(row["label"] is None for row in rows) == 384

for row in rows:
    assert list(row) == EXPECTED_FIELDS
    assert isinstance(row["id"], int)
    assert DOC_ID_RE.fullmatch(row["doc_id"])
    assert f"【{row['word']}】" in row["text"]
    assert isinstance(row["options"], str) and row["options"]
    assert row["label"] is None or isinstance(row["label"], str)
    assert isinstance(row["type"], list) and row["type"]

    parsed = parse_machine_options(row["options"])
    option_ids = [next(iter(entry)) for entry in parsed]
    if row["label"] is not None:
        assert row["label"] in option_ids
```

## 使用原则

1. 正式训练和评估默认使用当前发布版 `train.json`、`test.json`。
2. 机翻版仅用于机器基线复现、版本差异分析或数据来源追溯。
3. 读取 JSON 时从 `payload["rows"]` 取得记录，不要遍历顶层对象键。
4. 监督训练前显式排除或单独处理 384 条 `label = null` 的记录。
5. 不要把 `null` 自动替换为第一个候选项。
6. 不要按数组位置或 Excel 行号与当前发布版合并，只使用 `doc_id`。
7. 不要假设机翻版与当前发布版拥有相同的 `word_id` 集合或候选项结构。
8. 发布实验结果时明确写明使用的是“历史机翻版”还是“当前发布版”。
