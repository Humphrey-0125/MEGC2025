import json
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

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

# ---------- 主函数 ----------
# pred_file = "/home/d1/zwb/yhf/7_30/qwen2.5-vl-debug/v4-20250730-233728/checkpoint-230/infer_result/20250731-t0.jsonl"
# pred_file = "/home/d1/zwb/yhf/7_30/qwen2.5-vl-debug/v2-20250730-201008/checkpoint-230/infer_result/20250730-t0.jsonl"
# pred_file = "/home/d1/zwb/yhf/7_30/qwen2.5-vl-debug/v4-20250730-233728/checkpoint-230/infer_result/20250731-t0.jsonl"
pred_file = "/home/d1/zwb/yhf/result/Qwen2.5-VL-7B-Instruct/infer_result/20250731-150815.jsonl"
gt_file = "/home/d1/zwb/yhf/test_dataset.jsonl"
preds = load_answers_pred(pred_file)
gts = load_answers_gt(gt_file)
bleu_scores = []
rouge_scores = []

scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
smooth = SmoothingFunction().method1

for key in gts:
    if key not in preds:
        print(f"No prediction found for {key}")
        continue
    ref_answer = gts[key]
    pred_answer = preds[key]
    if not pred_answer.strip() or not ref_answer.strip():
        print(f"Skipping empty answer for {key}")
        continue
    # BLEU
    bleu = sentence_bleu([ref_answer.split()], pred_answer.split(), smoothing_function=smooth)
    bleu_scores.append(bleu)
    # ROUGE-1 (Recall)
    rouge = scorer.score(ref_answer, pred_answer)['rouge1'].recall
    rouge_scores.append(rouge)

# 输出平均分
if bleu_scores:
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_rouge = sum(rouge_scores) / len(rouge_scores)

    print(f"Average BLEU: {avg_bleu:.4f}")
    print(f"Average ROUGE-1 (recall): {avg_rouge:.4f}")
else:
    print("No valid matched predictions found.")

# pred_file ="/home/d1/zwb/yhf/7_30/qwen2.5-vl-debug/v0-20250730-011134/checkpoint-570/infer_result/20250730-114823.jsonl"
# gt_file = "/home/d1/zwb/yhf/test_dataset.jsonl"


# preds = load_jsonl_pred(pred_path)
# gts = load_jsonl_gt(gt_path)

# # 初始化分数统计
# bleu_scores = []
# rouge_scores = []

# scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
# smooth = SmoothingFunction().method1

# for pred, ref in zip(preds, refs):
#     pred_answer = pred['answer'].lower().strip()
#     ref_answer = ref['answer'].lower().strip()

#     # BLEU
#     bleu = sentence_bleu([ref_answer.split()], pred_answer.split(), smoothing_function=smooth)
#     bleu_scores.append(bleu)

#     # ROUGE-1
#     rouge = scorer.score(ref_answer, pred_answer)['rouge1'].recall
#     rouge_scores.append(rouge)

# # 输出平均结果
# avg_bleu = sum(bleu_scores) / len(bleu_scores)
# avg_rouge = sum(rouge_scores) / len(rouge_scores)

# print(f"Average BLEU: {avg_bleu:.4f}")
# print(f"Average ROUGE-1 (recall): {avg_rouge:.4f}")
