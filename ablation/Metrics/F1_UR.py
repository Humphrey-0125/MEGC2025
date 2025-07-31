import json
from sklearn.metrics import f1_score, classification_report
from collections import defaultdict, Counter

# ---------------- 配置文件路径 ----------------
# pred_file = "/home/d1/zwb/yhf/7_30/qwen2.5-vl-debug/v4-20250730-233728/checkpoint-230/infer_result/20250731-t0.jsonl"
pred_file = "/home/d1/zwb/yhf/result/Qwen2.5-VL-7B-Instruct/infer_result/20250731-150815.jsonl"
gt_file = "/home/d1/zwb/yhf/test_dataset.jsonl"
# ---------------- 工具函数 ----------------
def load_answers_gt(filepath):
    data = defaultdict(dict)  # key = (image_id, question) -> answer
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())  # 解析每行的JSON对象
            # 从messages中提取问题（user角色的content）
            messages = item["messages"]
            question = next(msg["content"] for msg in messages if msg["role"] == "user")
            # 从messages中提取回答（assistant角色的content）
            answer = next(msg["content"] for msg in messages if msg["role"] == "assistant")
            # 取第一张图片路径作为image_id（与原逻辑保持一致）
            image_id = item["images"][0]
            # 构建key并存储结果
            key = (image_id, question)
            data[key] = answer.strip().lower()
    return data

# def load_answers_gt_test_dataset_no(filepath):
#     data = defaultdict(dict)  # key = (image_id, question) -> answer
#     with open(filepath, "r", encoding="utf-8") as f:
#         for line in f:
#             item = json.loads(line.strip())  # 解析每行的JSON对象
#             # 从messages中提取问题（user角色的content）
#             messages = item["messages"]
#             question = next(msg["content"] for msg in messages if msg["role"] == "user")
#             # 从messages中提取回答（assistant角色的content）
#             answer = next(msg["content"] for msg in messages if msg["role"] == "assistant")
#             # 取第一张图片路径作为image_id（与原逻辑保持一致）
#             image_id = item["images"][0]
#             # 构建key并存储结果
#             key = (image_id, question)
#             data[key] = answer.strip().lower()
#     return data

def load_answers_pred(filepath):
    data = defaultdict(dict)  # key = (image_id, question) -> answer
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())  # 解析每行的JSON对象
            # 从messages中提取问题（user角色的content）
            messages = item["messages"]
            question = next(msg["content"] for msg in messages if msg["role"] == "user")          
            answer=item["response"]
            image_id = item["images"][0]["path"]  # 取第一张图片的路径作为image_id
            # 构建键值对
            key = (image_id, question)
            data[key] = answer.strip().lower()
    return data


def is_target_question(q):
    return q.lower() in {
        "what is the coarse expression class?",
        "what is the fine-grained expression class?"
    }

# ---------------- 主逻辑 ----------------
# 加载数据
pred_dict = load_answers_pred(pred_file)
gt_dict = load_answers_gt(gt_file)

# 对 coarse 和 fine 分开评估
categories = ["coarse", "fine"]
question_map = {
    "what is the coarse expression class?": "coarse",
    "what is the fine-grained expression class?": "fine"
}

eval_data = {
    "coarse": {"y_true": [], "y_pred": []},
    "fine": {"y_true": [], "y_pred": []}
}

# 遍历 ground truth
for (image_id, question), true_ans in gt_dict.items():
    if not is_target_question(question):
        # print(f"{image_id} - {question} 不是目标问题")
        continue
    if (image_id, question) not in pred_dict:
        print(f"缺少预测: {image_id} - {question}")
        continue
    pred_ans = pred_dict[(image_id, question)]
    label_type = question_map[question.lower()]
    eval_data[label_type]["y_true"].append(true_ans)
    eval_data[label_type]["y_pred"].append(pred_ans)

# ---------------- 评估函数 ----------------
# def evaluate_classification(y_true, y_pred, label_type):
#     print(f"\n [{label_type.upper()}] 分类评估结果:")
#     labels = sorted(set(y_true + y_pred))
#     report = classification_report(y_true, y_pred, labels=labels, digits=4, output_dict=True)
    
#     # Macro F1
#     macro_f1 = report["macro avg"]["f1-score"]
#     print(f"Macro-F1: {macro_f1:.4f}")

#     # Micro F1
#     micro_f1 = f1_score(y_true, y_pred, average='micro')
#     print(f"Micro-F1: {micro_f1:.4f}")

#     # UR1 (Unweighted Recall@1)
#     per_class_recall = [report[label]["recall"] for label in labels]
#     ur1 = sum(per_class_recall) / len(per_class_recall)
#     print(f"UR1 (Unweighted Recall@1): {ur1:.4f}")

def evaluate_classification(y_true, y_pred, label_type):
    # 过滤y_true或y_pred为None的样本
    valid = [(t, p) for t, p in zip(y_true, y_pred) if t is not None and p is not None]
    if not valid:
        print(f"[警告] 没有有效的 {label_type} 数据用于评估。")
        return
    y_true, y_pred = zip(*valid)  # 更新为过滤后的样本
    
    print(f"\n [{label_type.upper()}] 分类评估结果:")
    labels = sorted(set(y_true + y_pred))
    report = classification_report(y_true, y_pred, labels=labels, digits=4, output_dict=True)
    
    # Macro F1（即UF1）
    macro_f1 = report["macro avg"]["f1-score"]
    print(f"UF1 (Macro-F1): {macro_f1:.4f}")

    # UAR（即所有类别的召回率平均值）
    uar = report["macro avg"]["recall"]
    print(f"UAR (Unweighted Average Recall): {uar:.4f}")

    # Micro F1（保持不变）
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    print(f"Micro-F1: {micro_f1:.4f}")

# ---------------- 执行评估 ----------------
for cat in categories:
    evaluate_classification(eval_data[cat]["y_true"], eval_data[cat]["y_pred"], cat)
