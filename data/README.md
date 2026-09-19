# DynaSense 数据说明

本目录保存 DynaSense 当前使用的扁平化古汉语词义消歧（Word Sense Disambiguation, WSD）数据。

`train.json` 和 `test.json` 是正式数据文件。两者的每个元素都对应一个需要判断义项的目标字实例，合计 24,000 条。当前标签已经逐条复核并按批次确认。

本文件是 `data/` 目录当前数据结构的规范说明。项目中的部分旧注释、旧路径和 `LLM/LLM-guide.md` 仍展示过往的嵌套结构（例如顶层文档包含 `word_matches`），不能用来解释当前的 `train.json` 和 `test.json`。

## 数据版本导航

本目录现在同时保留两类数据，必须按标签来源区分：

| 版本 | 路径 | 标签来源 | 推荐用途 |
|---|---|---|---|
| 当前人工审核版 | `train.json`、`test.json` | 人工逐条复核 | 正式训练和评估 |
| 历史机翻版 | `machine_translation_original/` | 原机器生成结果，包含空标签 | 机器基线、误差分析和来源追溯 |

历史机翻版有独立的数据结构和使用限制，详见 [`machine_translation_original/README.md`](machine_translation_original/README.md)。它不是 `train.json`、`test.json` 的未切分副本，也不能替代当前人工审核金标准。

## 文件一览

| 文件 | 条目数 | 用途 | 当前状态 |
|---|---:|---|---|
| `train.json` | 19,200 | 正式训练集 | 可直接用于 PLM/LLM 训练 |
| `test.json` | 4,800 | 正式测试集 | 可直接用于评估 |
| `wsd_train.json` | 2,400 | 旧版 LoRA 数据占位文件 | 当前全部为空记录，不可直接训练 |
| `machine_translation_original/` | 24,000 | 独立保存的历史机翻基线 | 不属于正式训练/测试划分；先阅读子目录 README |

正式数据集只由 `train.json` 和 `test.json` 构成。两者采用固定的 80%/20% 划分，`doc_id` 没有交集。为了复现实验结果，请使用仓库给定的划分，不要重新随机切分后仍称为官方结果。

当前数据文件的 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `train.json` | `3f3c95f821612b4fbdaad3909840faa126737b711f0533f0a1170e7d7c9fea32` |
| `test.json` | `798454e1db87d86c90971562da170625835d15a94b6c98573525bb6af5cc0627` |

## 数据规模

| 指标 | 数值 |
|---|---:|
| 正式数据总条目数 | 24,000 |
| 唯一 `doc_id` 数 | 24,000 |
| 唯一来源文档 `id` 数 | 3,524 |
| 朝代数 | 12 |
| 唯一 `word_id` 数 | 63 |
| 唯一目标字数 | 63 |
| 同时属于两个 `type` 的条目数 | 1,928 |

训练集和测试集都覆盖全部 63 个 `word_id`。

## JSON 顶层结构

三个文件的顶层都是 JSON 数组。正式数据中的每个元素严格包含以下九个字段，并按下列顺序保存：

```text
id, doc_id, text, dynasty, word_id, word, options, label, type
```

结构示例（为便于阅读，`text` 已缩短）：

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

## 字段定义

| 字段 | JSON 类型 | 必填 | 含义与约束 |
|---|---|---|---|
| `id` | integer | 是 | 来源古文文档的数字编号。一个来源文档可以产生多条 WSD 数据，因此 `id` 不唯一，不能作为样本主键。 |
| `doc_id` | string | 是 | 当前 WSD 条目的唯一标识，格式为 `doc_<id>_<8位数字>`。24,000 条记录中全局唯一，是断点续跑、预测结果合并和去重时应使用的主键。 |
| `text` | string | 是 | 包含上下文的古文原文。需要消歧的目标字至少有一次以 `【word】` 形式标记。文本按 UTF-8 保存。 |
| `dynasty` | string | 是 | 原文所属朝代。它是朝代适配器和分朝代评估使用的字段。 |
| `word_id` | string | 是 | 目标字在义项表中的稳定标识，例如 `w10`、`w45-2`。模型分类头和义项表关联应使用 `word_id`，不要只使用字面上的 `word`。 |
| `word` | string | 是 | 当前需要消歧的目标字。当前版本有 63 个 `word_id`，每个 `word_id` 对应一个目标字。 |
| `options` | array<object> | 是 | 本条记录可选的义项。每个数组元素都是只含一个键值对的对象：键为义项 ID，值为义项释义。数组顺序有意义，不能排序或转为普通字典后再保存。 |
| `label` | string | 是 | 人工确认的正确义项 ID。它必须出现在本条记录的 `options` 中。`label` 不是朝代、不是从零开始的类别索引，也不是跨 `word_id` 的全局类别。 |
| `type` | array<object> | 是 | 目标字所属的统计类别。每个元素都是 `{type代码: type名称}`；一条记录可以有一个或两个类别。 |

## 唯一键与字段关系

### `doc_id` 才是条目主键

- `doc_id`：全局唯一，可用于合并预测结果。
- `id`：来源文档编号，会重复。
- `word_id`：目标字/义项表标识，会在许多样本中重复。
- `label`：只在对应 `word_id` 和当前 `options` 中有意义。

如果需要建立可跨记录比较的义项键，应至少使用 `(word_id, label)`，不能只使用 `label`。例如不同目标字都可能使用 `s1`，但这些 `s1` 的含义完全不同。

### `word_id` 与 `word`

当前数据中，63 个 `word_id` 分别对应 63 个目标字；二者是一对一关系。不过处理代码仍应保留 `word_id`，因为它承担义项表和多分类头的稳定标识作用，不能用 `word` 临时重建。

## `text` 中的目标标记

目标字至少有一次以如下形式出现：

```python
target_marker = f"【{row['word']}】"
assert target_marker in row["text"]
```

不要假设全文只有一对 `【】`。部分原文同时保留其他校注或引文标记；也有记录包含目标字的多个标记位置。因此：

- 定位目标时应精确查找 `【` + `word` + `】`；
- 不要把全文第一个 `【...】` 无条件当成目标字；
- 不要为了清洗文本而删除括号内的内容；
- 如果模型不需要标记，可以仅移除目标标记两侧的括号，但训练和测试必须使用同一规则。

## `options` 与 `label` 的正确解析

### 保持数组，不要直接转成字典

标准结构是：

```json
"options": [
  {"s1": "义项一"},
  {"s2": "义项二"}
]
```

不能使用下面这种写法把整个数组直接压成字典：

```python
# 不推荐：重复的义项 ID 会被覆盖
option_map = {
    option_id: meaning
    for entry in row["options"]
    for option_id, meaning in entry.items()
}
```

当前数据有 957 条记录在同一 `options` 数组中出现重复义项 ID，其中 42 条记录的 `label` 对应不止一个同名 ID。重复 ID 可能配有不同释义，所以必须保留数组顺序和全部元素。

安全的解析方式：

```python
def option_pairs(row):
    pairs = []
    for index, entry in enumerate(row["options"]):
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError(f"invalid option at {row['doc_id']}[{index}]")
        option_id, meaning = next(iter(entry.items()))
        pairs.append((index, option_id, meaning))
    return pairs


pairs = option_pairs(row)
label_matches = [pair for pair in pairs if pair[1] == row["label"]]
if not label_matches:
    raise ValueError(f"label not found: {row['doc_id']}")
```

当前 PLM 代码把 `label` 转为分类索引时采用第一个匹配项。若新代码也以数组索引作为类别，必须明确沿用这一规则；若任务需要区分重复 ID 对应的不同释义，则现有 `label` 本身不足以消除歧义，应保留完整候选数组并另行定义消歧策略。

### 不要假设义项 ID 只有一种格式

义项 ID 不只包含 `s1`、`s2`。数据中还会出现以下形式：

```text
1s1, 2s1, s0-1, s0-4, h0, h1, ...
```

因此不要使用只接受 `s\d+` 的正则表达式验证标签。应直接以当前记录 `options` 中实际出现的键为合法集合。

### 候选集是逐条记录字段

候选数范围为 3–35，平均约 14.519。即使 `word_id` 相同，候选集也不一定完全一致：当前 63 个 `word_id` 中有 38 个存在两种或三种 `options` 版本。因此不要只读取某个 `word_id` 的第一条记录，然后假定其候选数组适用于该字的全部样本。

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

22,072 条记录只有一个 `type`，1,928 条记录有两个 `type`。因此各类型记录数之和为 25,928，大于数据集总条目数 24,000。统计时应按多标签数据处理，不能假设六类互斥。

解析示例：

```python
type_pairs = [
    (code, name)
    for entry in row["type"]
    for code, name in entry.items()
]
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

朝代感知代码必须读取 `dynasty`。在当前扁平数据中：

```python
dynasty = row["dynasty"]
gold_sense_id = row["label"]
```

绝对不要把 `label` 当作朝代。`label` 的值类似 `s3`、`1s4` 或 `h0`，而 `dynasty` 的值才是“唐”“宋”等朝代名称。

## Python 加载与强校验示例

以下代码覆盖当前数据最重要的结构约束：

```python
import json
import re
from pathlib import Path


DATA_DIR = Path("data")
EXPECTED_FIELDS = [
    "id", "doc_id", "text", "dynasty", "word_id",
    "word", "options", "label", "type",
]
DOC_ID_RE = re.compile(r"^doc_\d+_\d{8}$")


def load_split(name: str) -> list[dict]:
    path = DATA_DIR / f"{name}.json"
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise TypeError(f"{path} must contain a JSON array")
    return rows


def validate_row(row: dict) -> None:
    if list(row) != EXPECTED_FIELDS:
        raise ValueError(f"field mismatch: {row.get('doc_id')}")
    if not isinstance(row["id"], int):
        raise TypeError(f"id is not int: {row['doc_id']}")
    if not DOC_ID_RE.fullmatch(row["doc_id"]):
        raise ValueError(f"invalid doc_id: {row['doc_id']}")
    if f"【{row['word']}】" not in row["text"]:
        raise ValueError(f"target marker missing: {row['doc_id']}")

    option_ids = []
    for entry in row["options"]:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError(f"invalid options: {row['doc_id']}")
        option_id, meaning = next(iter(entry.items()))
        if not isinstance(option_id, str) or not isinstance(meaning, str):
            raise TypeError(f"invalid option pair: {row['doc_id']}")
        option_ids.append(option_id)
    if row["label"] not in option_ids:
        raise ValueError(f"label not in options: {row['doc_id']}")

    if not isinstance(row["type"], list) or not row["type"]:
        raise ValueError(f"missing type: {row['doc_id']}")
    for entry in row["type"]:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError(f"invalid type: {row['doc_id']}")


train = load_split("train")
test = load_split("test")

assert len(train) == 19_200
assert len(test) == 4_800

for row in train + test:
    validate_row(row)

train_ids = {row["doc_id"] for row in train}
test_ids = {row["doc_id"] for row in test}
assert len(train_ids) == len(train)
assert len(test_ids) == len(test)
assert train_ids.isdisjoint(test_ids)
```

## 训练代码中的标签转换

原始 `label` 是义项 ID。PLM 多分类模型需要整数索引时，可以按本条记录的候选数组顺序转换：

```python
pairs = option_pairs(row)
matching_indices = [index for index, option_id, _ in pairs if option_id == row["label"]]
if not matching_indices:
    raise ValueError(f"unresolvable label: {row['doc_id']}")

# 与仓库当前 PLM 脚本的行为一致：重复 ID 时取第一个位置
label_index = matching_indices[0]
```

预测后将整数索引还原为义项 ID：

```python
predicted_entry = row["options"][predicted_index]
predicted_option_id = next(iter(predicted_entry))
```

不要把不同 `word_id` 的整数索引或 `label` 直接放进一个共享的全局类别空间，除非先显式建立新的全局映射。

## 与项目代码的对应关系

### PLM

- `PLM/3-bert-original.py` 读取 `train.json` 和 `test.json`，使用 `text` 与 `word` 作为 tokenizer 的句对输入。
- `PLM/3-bert-adapter.py` 使用 `text`、`word`、`dynasty` 和 `label` 生成朝代 adapter。
- `PLM/3-bert-ad-train.py` 和 `PLM/3-bert-cl-train.py` 同时使用 `word_id`、`options`、`label` 与 `dynasty`。
- 当前 PLM 数据集类在找不到 `label` 时会退回候选 0。正式数据不存在这种情况；新代码更适合直接报错，避免把坏数据静默标成第一义项。

### LLM

- `LLM/4-LLM-main.py` 面向当前扁平结构，主要读取 `doc_id`、`text`、`word` 和 `options`，并向结果中增加 `LLM_test_option_id`。
- `LLM/4-LLM-report.py` 读取的是模型预测结果，不是原始数据；它要求记录同时具有 `dynasty`、`label` 和 `LLM_test_option_id`。
- `LLM/4-LLM-adapter.py` 使用 `text`、`word`、`dynasty` 和 `label` 训练朝代 adapter。
- 使用 `LLM/4-LLM-main-integrated.py` 时，朝代必须从 `item["dynasty"]` 读取。该文件中兼容旧结构的 `item.get("label")` 回退逻辑不适用于当前扁平数据，因为当前 `label` 是正确义项 ID。
- 项目脚本中的数据和模型路径多为本机绝对路径，运行前需要改为实际仓库路径，例如 `data/train.json` 或 `data/test.json`。

### LoRA 与 `wsd_train.json`

当前 `data/wsd_train.json` 包含 2,400 条相同的空结构记录，所有字段均为 `null`、空字符串或空数组。它只是占位文件，不能作为有效训练数据。

`LLM/4-LLM-LoRA.ipynb` 中的数据生成单元仍引用旧字段 `final_option_id`，并把 `label` 当作分组标签。若要从当前扁平数据重新生成 LoRA 指令数据，应进行以下调整：

- 按需要使用 `dynasty` 分组或采样，而不是使用 `label`；
- 指令输出使用当前记录的 `label`；
- 输入使用 `text`、`word`，必要时加入格式化后的 `options`；
- 输出文件应采用 LLaMA-Factory 需要的 `instruction`、`input`、`output` 结构；
- 生成后先验证没有空记录，再用于训练。

## 预测结果字段约定

原始数据中的 `label` 是金标准，不应被模型预测覆盖。建议复制记录后添加独立预测字段：

| 模型流程 | 建议/现有预测字段 |
|---|---|
| PLM | `model_option_id` |
| LLM | `LLM_test_option_id` |

预测 ID 应当是本条 `options` 中已有的义项 ID；无法解析时可以写 `null`，但应单独统计，不要改写金标准 `label`。

## 当前数据质量约束

对 `train.json` 与 `test.json` 的全量检查结果：

- 24,000 条记录全部具有规定的九个字段；
- 24,000 个 `doc_id` 全部唯一，训练集和测试集无交集；
- 所有 `label` 都能在本条 `options` 中找到；
- 所有记录都包含精确目标标记 `【word】`；
- 所有 `options` 元素都是单键字符串对象；
- 所有记录都有非空 `type`，且 type 代码与名称保持一致；
- 所有 12 个朝代在训练集和测试集中都有覆盖；
- 全部 63 个 `word_id` 在训练集和测试集中都有覆盖。

本次人工复核版相对于仓库此前的机器标注版调整了 6,948 条 `label`，并为 4 条记录补充了审核所需的候选义项。上述变更已经同步到固定的训练集和测试集划分中。

## 预处理时必须保持的内容

1. 使用 UTF-8 读取和写入 JSON，避免古字和生僻字符损坏。
2. 保留 `doc_id`，它是结果合并和断点续跑的唯一键。
3. 保留 `options` 的数组顺序、重复 ID 和全部释义，不要去重。
4. 不要重编号 `label`；若模型需要整数类别，应在运行时建立映射。
5. 不要把 `label` 当作朝代，朝代字段始终是 `dynasty`。
6. 不要假设同一 `word_id` 的每条记录拥有完全相同的候选数组。
7. 不要把全文中任意一对 `【】` 都视为目标；应精确匹配 `【word】`。
8. 为了复现实验，保留仓库给出的训练集和测试集划分。
