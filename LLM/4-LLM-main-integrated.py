"""
4-LLM-main-integrated.py

改造版本：集成dynasty adapters的字词义项选择任务

核心改动:
  1. 初始化时加载LLMWithAdapterWrapper
  2. translate_single_word 函数添加dynasty参数
  3. process_item 从数据中提取dynasty并传递给模型调用
  4. 支持fallback: 如果adapter初始化失败, 降级使用原始LLM
  5. 适配新数据结构 (扁平化, 每条记录对应一个字词)

使用:
  python 4-LLM-main-integrated.py
"""

import os
import re
import json
import time
import logging
import threading
import torch
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from logging.handlers import RotatingFileHandler
from tqdm import tqdm

# 导入adapter wrapper
from unified_adapter import LLMWithAdapterWrapper


# ========================================
# 配置常量
# ========================================

# Xinference配置 (备用方案)
XINFERENCE_URL = "http://localhost:9997"
MODEL_UID = "qwen3"

# 模型和adapter配置
MODEL_NAME = "/root/autodl-tmp/models/Qwen3-1.5B-Instruct"
ADAPTER_PT_PATH = "/root/autodl-tmp/atd/data/结果/LLM/dynasty_adapters.pt"

# 数据和输出路径
JSON_PATH = "/root/autodl-tmp/atd/data/人工标注/最终结果/merged_dataset_Type.json"
OUTPUT_JSON = "/root/autodl-tmp/atd/data/结果/LLM/result_with_adapters.json"
LOG_FILE = "/root/autodl-tmp/atd/data/结果/LLM/result_with_adapters.log"

# 并发和超时配置
MAX_WORKERS = 100
MODEL_CALL_TIMEOUT = 6000000
MODEL_MAX_RETRIES = 3

# Adapter配置
USE_ADAPTER = True  # 是否使用adapter
ADAPTER_SCALE = 0.3  # adapter影响权重
ADAPTER_FALLBACK_TO_AVERAGE = True  # 如果朝代不在vocab中, 使用平均adapter

# ========================================
# 初始化 (日志、模型、adapter)
# ========================================

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
    logger = logging.getLogger("word_selector_adapter")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 文件处理器 (滚动)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)

    # 控制台 (与 tqdm 兼容)
    console_handler = TqdmLoggingHandler()
    console_handler.setFormatter(formatter)

    # 移除已有处理器
    if logger.handlers:
        for h in logger.handlers[:]:
            logger.removeHandler(h)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


logger = setup_logging()

# 初始化LLM + Adapter wrapper
llm_wrapper = None
use_adapter = False

try:
    if USE_ADAPTER:
        print(f"[初始化] 正在加载 LLM + Adapter wrapper...")
        llm_wrapper = LLMWithAdapterWrapper(
            model_name=MODEL_NAME,
            adapter_pt_path=ADAPTER_PT_PATH,
            device="cuda" if torch.cuda.is_available() else "cpu",
            adapter_scale=ADAPTER_SCALE
        )
        use_adapter = True
        info = llm_wrapper.get_adapter_info()
        logger.info(f"LLM + Adapter wrapper 初始化成功")
        logger.info(f"adapter信息: {info}")
except Exception as e:
    logger.warning(f"无法初始化 adapter wrapper: {e}")
    logger.info("将降级使用原始Xinference客户端")
    use_adapter = False
    llm_wrapper = None

# 初始化 Xinference 客户端 (备用)
client = None
model = None

if not use_adapter:
    try:
        from xinference.client import RESTfulClient
        client = RESTfulClient(XINFERENCE_URL)
        model = client.get_model(MODEL_UID)
        logger.info(f"连接 Xinference 成功: {XINFERENCE_URL}, 模型 {MODEL_UID}")
    except Exception as e:
        logger.warning(f"无法初始化 Xinference 客户端: {e}")
        client = None
        model = None


# ========================================
# 系统提示词
# ========================================

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


# ========================================
# 工具函数
# ========================================

def format_options_for_prompt(options_list):
    """将options列表格式化为字符串

    options_list格式示例: [{"s1": "逃跑, 逃亡"}, {"s2": "不在, 外出"}, ...]
    """
    lines = []
    if not options_list:
        return ""
    for opt in options_list:
        if isinstance(opt, dict):
            for k, v in opt.items():
                lines.append(f"{k}: {v}")
        elif isinstance(opt, str):
            lines.append(opt)
    return "\n".join(lines)


def extract_json_from_text(text):
    """从文本中抽取JSON内容"""
    if not isinstance(text, str):
        return None

    t = text.strip()
    t = re.sub(r'```json\s*', '', t, flags=re.IGNORECASE)
    t = re.sub(r'```', '', t)

    # 尝试直接解析
    try:
        return json.loads(t)
    except Exception:
        pass

    # 尝试提取 { ... }
    m = re.search(r'(\{[\s\S]*\})', t)
    if m:
        candidate = m.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            cand2 = candidate.replace("'", '"')
            cand2 = cand2.replace("，", ",").replace(": ", ":")
            try:
                return json.loads(cand2)
            except Exception:
                return None

    return None


def parse_model_answer(raw_text):
    """从模型返回文本中解析sense_id和confidence"""
    if not raw_text:
        return None, 0.0

    # 尝试抽取JSON
    parsed = extract_json_from_text(raw_text)
    if isinstance(parsed, dict):
        sense_id = parsed.get("sense_id")
        confidence = parsed.get("confidence")

        if isinstance(sense_id, str) and sense_id.lower() == "null":
            sense_id = None

        try:
            confidence = float(confidence) if confidence is not None else 0.0
        except Exception:
            confidence = 0.0

        return sense_id, confidence

    # 尝试找到 "s\d+" 或 "null"
    m = re.search(r'\b(s\d+)\b', raw_text)
    if m:
        return m.group(1), 1.0

    if re.search(r'\bnull\b', raw_text, flags=re.IGNORECASE):
        return None, 0.0

    return None, 0.0


# ========================================
# 模型调用函数
# ========================================

def call_model(messages, timeout=MODEL_CALL_TIMEOUT, dynasty=None):
    """
    调用LLM进行推理 (优先使用adapter wrapper)

    Args:
        messages: 标准OpenAI格式的消息列表
        timeout: 超时时间
        dynasty: 朝代标签 (用于adapter选择)

    Returns:
        模型的回复文本
    """
    if use_adapter and llm_wrapper is not None:
        # 使用adapter wrapper进行推理
        try:
            # 从messages中提取prompt
            system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
            user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")

            # 组合prompt
            full_prompt = f"{system_msg}\n\n{user_msg}" if system_msg else user_msg

            # 调用带adapter的生成
            reply = llm_wrapper.generate(
                text=full_prompt,
                dynasty=dynasty if dynasty else None,
                max_length=2048,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

            return reply

        except Exception as e:
            logger.error(f"Adapter wrapper推理失败: {e}")
            raise

    elif client is not None and model is not None:
        # 降级: 使用Xinference客户端
        try:
            with ThreadPoolExecutor(max_workers=1) as exe:
                future = exe.submit(
                    model.chat,
                    messages,
                    generate_config={"temperature": 0.7, "max_tokens": 2048}
                )
                completion = future.result(timeout=timeout)
                reply = completion["choices"][0]["message"]["content"]
                return reply
        except TimeoutError:
            raise TimeoutError("Xinference请求超时")
        except Exception as e:
            logger.error(f"Xinference调用失败: {e}")
            raise

    else:
        raise RuntimeError("没有可用的模型后端 (既未初始化adapter wrapper, 也未初始化Xinference)")


# ========================================
# 核心处理函数
# ========================================

def translate_single_word(idx, text, word, options_list,
                         dynasty=None, max_retries=MODEL_MAX_RETRIES):
    """
    调用大模型判断word对应哪个义项

    Args:
        idx: 记录索引 (用于日志)
        text: 古文原文
        word: 待匹配字词
        options_list: 义项列表
        dynasty: 朝代标签 (用于adapter选择)
        max_retries: 重试次数

    Returns:
        sense_id (str 或 None)
    """
    # 构建prompt
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
            # 调用模型 (传入dynasty)
            reply_text = call_model(messages, timeout=MODEL_CALL_TIMEOUT, dynasty=dynasty)
            sense_id, conf = parse_model_answer(reply_text)

            # 如果sense_id无效, 重试
            if sense_id is None or str(sense_id).lower() in ("null", ""):
                logger.warning(
                    f"模型无有效返回 (idx={idx}, sense_id=null), 尝试 {attempt}/{max_retries} - word={word}"
                )
                time.sleep(2 ** (attempt - 1))
                continue

            # 成功
            logger.info(
                f"模型返回: idx={idx}, word={word}, dynasty={dynasty}, sense_id={sense_id}, confidence={conf}"
            )
            return sense_id

        except TimeoutError as te:
            last_err = te
            logger.warning(f"模型调用超时, 尝试 {attempt}/{max_retries} - word={word}")
            time.sleep(2 ** (attempt - 1))
        except Exception as e:
            last_err = e
            logger.error(f"模型调用失败: {e} ; 尝试 {attempt}/{max_retries}")
            time.sleep(2 ** (attempt - 1))

    logger.error(f"模型调用最终失败: word={word} ; 错误: {last_err}")
    return None


def process_item(item):
    """
    处理单条记录 (新数据结构：每条记录对应一个字词)

    从item中提取dynasty信息, 调用模型得到义项ID, 添加LLM_test_option_id字段
    """
    idx = item.get("doc_id", -1)
    text = item.get("text", "") or ""
    word = item.get("word", "") or ""
    options_list = item.get("options", []) or []

    # 检查是否已有LLM_test_option_id, 若有则跳过
    if "LLM_test_option_id" in item and item["LLM_test_option_id"] is not None:
        logger.info(f"跳过已处理的记录: idx={idx}, word={word}, LLM_test_option_id={item['LLM_test_option_id']}")
        return item

    # 从item中提取dynasty (新数据中朝代字段为label)
    dynasty = item.get("label") or item.get("dynasty") or item.get("period") or None

    logger.info(f"处理记录: idx={idx}, word={word}, dynasty={dynasty}, options数量={len(options_list)}")

    # 调用模型获取义项ID
    sense_id = translate_single_word(idx, text, word, options_list, dynasty=dynasty)

    # 添加LLM_test_option_id字段
    item["LLM_test_option_id"] = sense_id if sense_id is not None else None

    logger.info(f"完成记录: idx={idx}, word={word}, 结果={sense_id}")

    return item


# ========================================
# 保存/合并函数
# ========================================

def save_results(results, output_path):
    """
    将结果保存/合并到输出文件
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    except Exception:
        pass

    with file_lock:
        existing_data = []
        if os.path.exists(output_path):
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except Exception as e:
                logger.warning(f"读取现有输出文件失败 (将覆盖): {e}")
                existing_data = []

        # 按doc_id合并 (doc_id在新数据中是唯一的)
        index_map = {it.get("doc_id", -1): it for it in existing_data if isinstance(it, dict)}
        for res in results:
            index_map[res.get("doc_id", -1)] = res

        # 排序并保存
        updated = [index_map[k] for k in sorted(index_map.keys())]
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(updated, f, ensure_ascii=False, indent=2)
            logger.info(f"保存 {len(results)} 条记录到 {output_path} (合并后总条目 {len(updated)})")
        except Exception as e:
            logger.error(f"写入输出文件失败: {e}")


# ========================================
# 主流程
# ========================================

def process_dataset(input_path=JSON_PATH, output_path=OUTPUT_JSON, max_workers=MAX_WORKERS):
    logger.info(f"开始处理数据集: {input_path}")
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        logger.info(f"读取输入数据集成功, 共 {len(dataset)} 条记录")
    except Exception as e:
        logger.error(f"读取输入文件失败: {e}")
        return []

    # 读取现有输出 (断点续跑)
    existing_indices = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                # 使用doc_id作为唯一标识 (因为新数据中doc_id唯一)
                existing_indices = {it.get("doc_id") for it in existing_data if isinstance(it, dict)}
            logger.info(f"发现已有输出文件, 包含 {len(existing_indices)} 条记录, 将跳过这些记录")
        except Exception as e:
            logger.warning(f"读取现有输出失败: {e}")

    # 过滤需要处理的记录
    items_to_process = [it for it in dataset if it.get("doc_id") not in existing_indices]
    if not items_to_process:
        logger.info("没有需要处理的新记录。")
        return []

    logger.info(f"需要处理 {len(items_to_process)} 条新记录 (并发 workers={max_workers})")

    all_results = []
    batch_size = 50
    for start in range(0, len(items_to_process), batch_size):
        batch = items_to_process[start:start + batch_size]
        logger.info(f"处理批次 {start//batch_size + 1}, 包含 {len(batch)} 条")
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
                        # 保留原始数据, 标记错误状态
                        err_item = item.copy()
                        err_item["LLM_test_option_id"] = None
                        err_item["state"] = "error"
                        batch_results.append(err_item)
                    pbar.update(1)

        if batch_results:
            save_results(batch_results, output_path)
            all_results.extend(batch_results)
        logger.info(f"完成批次 {start//batch_size + 1}")

    logger.info(f"全部处理完成, 共处理 {len(all_results)} 条记录")
    return all_results


# ========================================
# 脚本入口
# ========================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("开始字词选择任务 (集成adapter版本)")
    logger.info(f"输入: {JSON_PATH}")
    logger.info(f"输出: {OUTPUT_JSON}")
    logger.info(f"使用adapter: {use_adapter}")
    if use_adapter and llm_wrapper:
        logger.info(f"adapter朝代: {llm_wrapper.get_available_dynasties()}")
    logger.info("=" * 60)

    try:
        results = process_dataset()
        logger.info("任务完成。")
    except Exception as e:
        logger.exception(f"任务异常终止: {e}")