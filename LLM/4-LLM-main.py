"""
word_option_selector.py

用途:
    - 读取输入 JSON 数据集（每条包含 text、word、options 等字段）
    - 对于每条记录，调用大模型判断对应哪个义项（返回义项ID，如"s1"）
    - 将结果写入顶层字段 "LLM_test_option_id"（若无法匹配为 null）
    - 将处理好的记录保存到输出 JSON（支持断点续跑 / 合并）

使用:
    - 修改顶部配置项（JSON_PATH, OUTPUT_JSON, Xinference 地址 & model_uid）
"""

import os
import re
import json
import time
import logging
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from logging.handlers import RotatingFileHandler
from xinference.client import RESTfulClient
from tqdm import tqdm


# Xinference 配置 - 请按实际情况修改
XINFERENCE_URL = "http://localhost:9997"
MODEL_UID = "qwen3"

# -----------------------------
# 配置常量（请按实际路径/需求修改）
# -----------------------------

JSON_PATH = f"/root/autodl-tmp/atd/data/人工标注/最终结果/merged_dataset_Type.json"  # 输入JSON文件路径
OUTPUT_JSON = f"/root/autodl-tmp/atd/data/结果/LLM/{MODEL_UID}.json"  # 测试结果JSON文件
LOG_FILE = f"/root/autodl-tmp/atd/data/结果/LLM/{MODEL_UID}.log"

# 并发 & 超时配置
MAX_WORKERS = 100
MODEL_CALL_TIMEOUT = 6000000  # 模型调用等待秒数（单个调用）
MODEL_MAX_RETRIES = 3

# -----------------------------
# 初始化（日志 / 文件锁 / 模型客户端）
# -----------------------------
file_lock = threading.Lock()


class TqdmLoggingHandler(logging.Handler):
    """确保 logging 与 tqdm 进度条兼容"""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, end='\n')
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging(log_file=LOG_FILE):
    logger = logging.getLogger("word_selector")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 文件处理器（滚动）
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)

    # 控制台（与 tqdm 兼容）
    console_handler = TqdmLoggingHandler()
    console_handler.setFormatter(formatter)

    # 移除已有处理器，避免重复
    if logger.handlers:
        for h in logger.handlers[:]:
            logger.removeHandler(h)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


logger = setup_logging()

# 初始化 Xinference 客户端（如果可用）
try:
    client = RESTfulClient(XINFERENCE_URL)
    model = client.get_model(MODEL_UID)
    logger.info(f"连接 Xinference 成功: {XINFERENCE_URL}, 模型 {MODEL_UID}")
except Exception as e:
    logger.warning(f"无法初始化 Xinference 客户端（请检查配置）。异常: {e}")
    client = None
    model = None

# -----------------------------
# 系统提示词（发给大模型的 system prompt）
# -----------------------------
SYSTEM_PROMPT = """
你是一位专业的古汉语语言学专家, 擅长分析古文字词的含义和义项匹配。请根据提供的古文原文、待匹配字词, 结合该字词的所有义项, 选择匹配的义项id。

任务要求: 
1. 仔细分析古文原文上下文语境
2. 准确判断待匹配字词在文中的具体含义
3. 将模型给出的选择与提供的义项列表进行精确匹配
4. 如果选择直接匹配某个义项, 返回该义项的ID
5. 如果选择与某个义项是近义词关系, 返回该义项的ID
6. 如果以上都不成立，但选择在上下文中合理，请评估是否需要新增义项
7. 如果选择与任何义项都不匹配且不合理, 返回null

输出格式 (严格 JSON, 禁止包含多余文本): 
{"sense_id": "义项ID 或 null", "confidence": 0.0-1.0 之间的数值"}
"""

# -----------------------------
# 工具函数: 构建选项字符串 / 解析模型响应
# -----------------------------
def format_options_for_prompt(options_list):
    """
    将输入数据里的 options（例如 [{"s1": "解释1"}, {"s2": "解释2"}]）格式化成
    "s1: 解释1\ns2: 解释2\n..." 的字符串，便于放入模型 prompt。
    """
    lines = []
    if not options_list:
        return ""
    for opt in options_list:
        # opt 可能是 dict 或其他，尝试解析
        if isinstance(opt, dict):
            for k, v in opt.items():
                lines.append(f"{k}: {v}")
        elif isinstance(opt, str):
            # 万一是纯字符串（不常见），直接加入
            lines.append(opt)
    return "\n".join(lines)


def extract_json_from_text(text):
    """
    尝试从模型返回的文本中抽取 JSON 内容（优先），若失败返回 None。
    支持带 ```json``` 包裹、纯 json、或文本中包含 {...} 的情况。
    """
    if not isinstance(text, str):
        return None

    # 去掉开头/结尾的 code block 标记
    t = text.strip()
    t = re.sub(r'```json\s*', '', t, flags=re.IGNORECASE)
    t = re.sub(r'```', '', t)

    # 尝试直接解析整段为 JSON
    try:
        return json.loads(t)
    except Exception:
        pass

    # 尝试提取第一个 { ... } 块
    m = re.search(r'(\{[\s\S]*\})', t)
    if m:
        candidate = m.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            # 有时候模型用单引号或中文符号，做一步替换再试
            cand2 = candidate.replace("'", '"')
            cand2 = cand2.replace("，", ",").replace(": ", ":")
            try:
                return json.loads(cand2)
            except Exception:
                return None

    return None


def parse_model_answer(raw_text):
    """
    从模型返回文本中解析出 (sense_id, confidence):
      - 如果能抽出 JSON 且包含 sense_id/confidence，则返回对应值
      - 否则尝试匹配类似 s1/s2 的模式
      - 最后失败则返回 (None, 0.0)
    """
    if not raw_text:
        return None, 0.0

    # 先尝试抽取 JSON
    parsed = extract_json_from_text(raw_text)
    if isinstance(parsed, dict):
        sense_id = parsed.get("sense_id")
        confidence = parsed.get("confidence")
        # 规范化 null/'null' -> None
        if isinstance(sense_id, str) and sense_id.lower() == "null":
            sense_id = None
        try:
            confidence = float(confidence) if confidence is not None else 0.0
        except Exception:
            confidence = 0.0
        return sense_id, confidence

    # 如果没抽出 JSON，尝试找到 "s\d+" 或 "null" 的文本
    m = re.search(r'\b(s\d+)\b', raw_text)
    if m:
        return m.group(1), 1.0

    if re.search(r'\bnull\b', raw_text, flags=re.IGNORECASE):
        return None, 0.0

    # 无法解析
    return None, 0.0

# -----------------------------
# 模型调用函数（可替换为你实际的模型客户端）
# -----------------------------
def call_model(messages, timeout=MODEL_CALL_TIMEOUT):
    """
    这里只是一个示例封装。你需要把这个函数替换为实际模型的调用逻辑，
    返回值必须包含文本字符串（模型的回复）。
    示例（注释形式）:
        completion = model.chat(messages, generate_config={"temperature":0.0, "max_tokens": 512})
        reply_text = completion["choices"][0]["message"]["content"]
        return reply_text

    当前示例会抛出 NotImplementedError，提醒用户替换。
    """
    # ---------- 示例（请根据实际模型替换） ----------
    # 下面注释展示如何调用你原模板的 model.chat（若已初始化 model）

    with ThreadPoolExecutor(max_workers=1) as exe:
        future = exe.submit(model.chat, messages, generate_config={"temperature": 0.5, "max_tokens": 2048})
        completion = future.result(timeout=timeout)
        reply = completion["choices"][0]["message"]["content"]
        return reply

    # raise NotImplementedError(
    #     "call_model() 未实现。请将此函数替换为你实际的模型调用逻辑（例如调用 Xinference 的 model.chat）。"
    # )

# -----------------------------
# 核心: 为一条记录调用模型并返回 sense_id
# -----------------------------
def translate_single_word(idx, text, word, options_list, max_retries=MODEL_MAX_RETRIES):
    """
    调用大模型判断 word 对应哪个义项（options_list），
    返回 sense_id (str 或 None)
    """
    # 构建 user_prompt（把 options 格式化进去）
    options_str = format_options_for_prompt(options_list)
    user_prompt = (
        f'古文: "{text}"\n'
        f'待匹配字词: {word}\n'
        f'该字词所有义项(格式: 义项ID: 义项解释): \n{options_str}\n\n'
        "请以如下格式返回: {\"sense_id\": \"义项ID或null\", \"confidence\": 0.0-1.0之间的小数}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            reply_text = call_model(messages, timeout=MODEL_CALL_TIMEOUT)
            sense_id, conf = parse_model_answer(reply_text)
            # 若 sense_id 无效，则视为失败 -> 继续重试
            if sense_id is None or str(sense_id).lower() in ("null", ""):
                logger.warning(
                    f"模型无有效返回（idx={idx}, sense_id=null）, 尝试 {attempt}/{max_retries} - word={word}"
                )
                time.sleep(2 ** (attempt - 1))
                continue

            # 正常情况
            logger.info(
                f"模型返回: idx={idx}, word={word}, sense_id={sense_id}, confidence={conf}"
            )
            return sense_id
        except TimeoutError as te:
            last_err = te
            logger.warning(f"模型调用超时, 尝试 {attempt}/{max_retries} - word={word}")
            time.sleep(2 ** (attempt - 1))
        except NotImplementedError:
            # 在测试环境/未配置模型时，抛出 NotImplementedError，外层应捕获并中止
            raise
        except Exception as e:
            last_err = e
            logger.error(f"模型调用失败: {e} ; 尝试 {attempt}/{max_retries}")
            time.sleep(2 ** (attempt - 1))

    logger.error(f"模型调用最终失败: word={word} ; 错误: {last_err}")
    return None

# -----------------------------
# 处理单条记录（document）
# 新数据结构：每条记录直接包含 text, word, options 等字段，没有 word_matches 嵌套
# -----------------------------
def process_item(item):
    """
    item 应包含至少:
      - "doc_id" （用于合并）
      - "text" （古文）
      - "word" （待匹配字词）
      - "options" （列表，元素为 {"sX": "释义"}）
    对每条记录，调用模型并将结果存入 "LLM_test_option_id" 字段。
    返回更新后的 item（保留所有原有字段）。
    """
    idx = item.get("doc_id", -1)
    text = item.get("text", "") or ""
    word = item.get("word", "")
    options_list = item.get("options", []) or []

    # 如果已经存在 LLM_test_option_id 且不为 None，则跳过（支持断点续跑，但通常由外层跳过）
    if "LLM_test_option_id" in item and item["LLM_test_option_id"] is not None:
        logger.info(f"跳过已存在的记录: index={idx}, word={word}, LLM_test_option_id={item['LLM_test_option_id']}")
        return item

    logger.info(f"处理 index={idx}, word={word}, options数量={len(options_list)}")

    # 调用模型
    sense_id = translate_single_word(idx, text, word, options_list)
    item["LLM_test_option_id"] = sense_id if sense_id is not None else None

    return item


# -----------------------------
# 保存/合并输出函数
# -----------------------------
def save_results(results, output_path):
    """
    将一批结果保存到 output_path。
    若 output_path 已存在，则按 doc_id 合并/更新（以 doc_id 为唯一键）。
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    except Exception:
        # 目录可能是当前目录
        pass

    with file_lock:
        existing_data = []
        if os.path.exists(output_path):
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except Exception as e:
                logger.warning(f"读取现有输出文件失败（将覆盖）: {e}")
                existing_data = []

        # index -> item 映射
        index_map = {it.get("doc_id", -1): it for it in existing_data if isinstance(it, dict)}

        for res in results:
            index_map[res.get("doc_id", -1)] = res

        # 按 index 排序写回
        updated = [index_map[k] for k in sorted(index_map.keys())]
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(updated, f, ensure_ascii=False, indent=2)
            logger.info(f"保存 {len(results)} 条记录到 {output_path}（合并后总条目 {len(updated)}）")
        except Exception as e:
            logger.error(f"写入输出文件失败: {e}")


# -----------------------------
# 主流程: 处理整个数据集
# -----------------------------
def process_dataset(input_path=JSON_PATH, output_path=OUTPUT_JSON, max_workers=MAX_WORKERS):
    logger.info(f"开始处理数据集: {input_path}")
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        logger.info(f"读取输入数据集成功，共 {len(dataset)} 条记录")
    except Exception as e:
        logger.error(f"读取输入文件失败: {e}")
        return []

    # 读取现有输出（用于断点续跑）
    existing_indices = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                existing_indices = {it.get("doc_id") for it in existing_data if isinstance(it, dict)}
            logger.info(f"发现已有输出文件，包含 {len(existing_indices)} 条记录，将跳过这些记录")
        except Exception as e:
            logger.warning(f"读取现有输出失败: {e}")

    # 过滤需要处理的记录（根据 index）
    items_to_process = [it for it in dataset if it.get("doc_id") not in existing_indices]
    if not items_to_process:
        logger.info("没有需要处理的新记录。")
        return []

    logger.info(f"需要处理 {len(items_to_process)} 条新记录（并发 workers={max_workers}）")

    all_results = []
    batch_size = 50
    for start in range(0, len(items_to_process), batch_size):
        batch = items_to_process[start:start + batch_size]
        logger.info(f"处理批次 {start//batch_size + 1}，包含 {len(batch)} 条")
        batch_results = []

        with ThreadPoolExecutor(max_workers=min(max_workers, len(batch))) as executor:
            future_to_item = {executor.submit(process_item, item): item for item in batch}
            with tqdm(total=len(batch), desc=f"批次 {start//batch_size + 1}") as pbar:
                for future in concurrent.futures.as_completed(future_to_item):
                    item = future_to_item[future]
                    try:
                        res = future.result()
                        batch_results.append(res)
                    except Exception as e:
                        logger.exception(f"处理记录 index={item.get('doc_id')} 失败: {e}")
                        # 生成错误占位，保留原字段以便排查
                        err_item = item.copy()  # 保留原始字段
                        err_item["LLM_test_option_id"] = None
                        err_item["state"] = "error"
                        batch_results.append(err_item)
                    pbar.update(1)

        if batch_results:
            save_results(batch_results, output_path)
            all_results.extend(batch_results)
        logger.info(f"完成批次 {start//batch_size + 1}")

    logger.info(f"全部处理完成，共处理 {len(all_results)} 条记录")
    return all_results


# -----------------------------
# 脚本入口
# -----------------------------
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("开始字词选择任务")
    logger.info(f"输入: {JSON_PATH}")
    logger.info(f"输出: {OUTPUT_JSON}")
    logger.info("=" * 60)

    try:
        results = process_dataset()
        logger.info("任务完成。")
    except NotImplementedError:
        logger.error("模型调用函数尚未实现（call_model）。请实现该函数后重试。")
    except Exception as e:
        logger.exception(f"任务异常终止: {e}")