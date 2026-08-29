"""
evaluate_by_dynasty.py

功能：
    - 读取 JSON 数据集（每条包含 dynasty, label, LLM_test_option_id）
    - 按朝代（dynasty）输出 classification report（针对义项 ID）
    - 计算所有朝代的平均 macro avg 指标（precision, recall, f1-score）
    - 将结果保存到 Excel（每个朝代一个 sheet，另加 Average sheet）
    - 同时在控制台打印所有报告

使用：
    直接修改下方的 INPUT_JSON 和 OUTPUT_EXCEL 路径，然后运行
"""

import json
from collections import defaultdict
import pandas as pd
from sklearn.metrics import classification_report

# ==================== 配置区域 ====================
INPUT_JSON = "/root/autodl-tmp/atd/data/结果/LLM-test.json"  # 请修改为LLM 输出测试结果的路径
OUTPUT_EXCEL = "/root/autodl-tmp/atd/data/结果/LLM-excel.xlsx"  # 修改为实际输出 Excel 路径
# =================================================


def load_data(json_path):
    """加载 JSON 数据，确保每条记录包含所需字段"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    required_fields = {'dynasty', 'label', 'LLM_test_option_id'}
    valid_records = []
    for rec in data:
        if all(field in rec for field in required_fields):
            valid_records.append(rec)
        else:
            print(f"警告：跳过缺失字段的记录 {rec.get('id', rec.get('index', 'unknown'))}")
    return valid_records


def compute_dynasty_report(data):
    """按朝代分组，返回每个朝代的 classification report (dict) 和 macro avg 指标"""
    dynasty_groups = defaultdict(lambda: {'y_true': [], 'y_pred': []})

    for rec in data:
        dynasty = rec['dynasty']
        y_true = rec['label']
        y_pred = rec['LLM_test_option_id']
        # 确保标签非空
        if y_true is not None and y_pred is not None:
            dynasty_groups[dynasty]['y_true'].append(y_true)
            dynasty_groups[dynasty]['y_pred'].append(y_pred)

    dynasty_reports = {}
    dynasty_macros = {}  # 存储每个朝代的 macro avg 指标

    for dynasty, values in dynasty_groups.items():
        y_true = values['y_true']
        y_pred = values['y_pred']
        if len(set(y_true)) < 2:
            # 只有一个类别时 classification_report 可能警告，但仍可计算
            print(f"警告：朝代 {dynasty} 只有一个类别，报告可能不完整")
        report_dict = classification_report(
            y_true, y_pred,
            output_dict=True,
            zero_division=0
        )
        dynasty_reports[dynasty] = report_dict
        # 提取 macro avg 指标
        macro = report_dict.get('macro avg', {})
        dynasty_macros[dynasty] = {
            'precision': macro.get('precision', 0.0),
            'recall': macro.get('recall', 0.0),
            'f1-score': macro.get('f1-score', 0.0),
            'support': sum(report_dict.get(k, {}).get('support', 0)
                           for k in report_dict if k not in ['accuracy', 'macro avg', 'weighted avg'])
        }

    return dynasty_reports, dynasty_macros


def compute_average_macro(dynasty_macros):
    """计算所有朝代的 macro avg 指标的平均值"""
    if not dynasty_macros:
        return {}

    avg_precision = sum(m['precision'] for m in dynasty_macros.values()) / len(dynasty_macros)
    avg_recall = sum(m['recall'] for m in dynasty_macros.values()) / len(dynasty_macros)
    avg_f1 = sum(m['f1-score'] for m in dynasty_macros.values()) / len(dynasty_macros)
    total_support = sum(m['support'] for m in dynasty_macros.values())

    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1-score': avg_f1,
        'support': total_support,
        'num_dynasties': len(dynasty_macros)
    }


def report_dict_to_dataframe(report_dict):
    """将 classification_report 返回的 dict 转换为 DataFrame（方便 Excel 写入）"""
    # 移除 'accuracy' 键，它不在类别列表中
    rows = []
    for label, metrics in report_dict.items():
        if label in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        rows.append({
            'class': label,
            'precision': metrics.get('precision', 0.0),
            'recall': metrics.get('recall', 0.0),
            'f1-score': metrics.get('f1-score', 0.0),
            'support': metrics.get('support', 0)
        })
    # 添加 macro avg 和 weighted avg 行
    for avg_type in ['macro avg', 'weighted avg']:
        if avg_type in report_dict:
            metrics = report_dict[avg_type]
            rows.append({
                'class': avg_type,
                'precision': metrics.get('precision', 0.0),
                'recall': metrics.get('recall', 0.0),
                'f1-score': metrics.get('f1-score', 0.0),
                'support': metrics.get('support', 0)
            })
    # 添加 accuracy 行
    if 'accuracy' in report_dict:
        rows.append({
            'class': 'accuracy',
            'precision': None,
            'recall': None,
            'f1-score': report_dict['accuracy'],
            'support': None
        })
    return pd.DataFrame(rows)


def print_report(dynasty, report_dict):
    """控制台打印单个朝代的报告（简洁格式）"""
    print(f"\n{'='*60}")
    print(f"朝代：{dynasty}")
    print('='*60)
    # 直接使用 sklearn 的文本输出更清晰
    # 但我们有 report_dict，可以重新生成文本
    # 简单方法：从 report_dict 构建打印
    print(f"{'':>15} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
    for label, metrics in report_dict.items():
        if label in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        print(f"{label:>15} {metrics['precision']:10.4f} {metrics['recall']:10.4f} "
              f"{metrics['f1-score']:10.4f} {metrics['support']:10.0f}")
    # 打印 macro/weighted avg
    for avg_type in ['macro avg', 'weighted avg']:
        if avg_type in report_dict:
            m = report_dict[avg_type]
            print(f"{avg_type:>15} {m['precision']:10.4f} {m['recall']:10.4f} "
                  f"{m['f1-score']:10.4f} {m['support']:10.0f}")
    if 'accuracy' in report_dict:
        print(f"{'accuracy':>15} {'':>10} {'':>10} {report_dict['accuracy']:10.4f} {'':>10}")


def save_to_excel(dynasty_reports, average_macro, output_path):
    """将各朝代报告和平均结果保存到 Excel"""
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # 每个朝代一个 sheet
        for dynasty, report_dict in dynasty_reports.items():
            df = report_dict_to_dataframe(report_dict)
            # sheet 名称限制 31 字符，朝代名称可能过长，做截断
            sheet_name = dynasty[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        # 平均结果 sheet
        avg_df = pd.DataFrame([average_macro])
        avg_df.to_excel(writer, sheet_name='Average_Macro', index=False)

    print(f"\n结果已保存至 Excel: {output_path}")


def main():
    # 加载数据
    print(f"加载数据: {INPUT_JSON}")
    data = load_data(INPUT_JSON)
    print(f"成功加载 {len(data)} 条有效记录")

    # 按朝代计算报告
    dynasty_reports, dynasty_macros = compute_dynasty_report(data)
    print(f"发现 {len(dynasty_reports)} 个朝代")

    # 计算平均 macro 指标
    avg_macro = compute_average_macro(dynasty_macros)

    # 控制台输出
    print("\n" + "="*80)
    print("分类报告（按朝代）")
    print("="*80)
    for dynasty, report_dict in dynasty_reports.items():
        print_report(dynasty, report_dict)

    # 输出平均结果
    print("\n" + "="*60)
    print("所有朝代平均 Macro 指标")
    print("="*60)
    print(f"平均 Precision: {avg_macro['precision']:.4f}")
    print(f"平均 Recall:    {avg_macro['recall']:.4f}")
    print(f"平均 F1-score:  {avg_macro['f1-score']:.4f}")
    print(f"总样本数:       {avg_macro['support']:.0f}")
    print(f"参与朝代数量:   {avg_macro['num_dynasties']}")

    # 保存 Excel
    save_to_excel(dynasty_reports, avg_macro, OUTPUT_EXCEL)


if __name__ == '__main__':
    main()