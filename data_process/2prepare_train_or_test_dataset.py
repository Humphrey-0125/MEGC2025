import json
from collections import defaultdict
import os
import glob


jsonl_path = "/home/d1/zwb/yhf/data_json/casme_train.jsonl"
image_root = "/home/d1/zwb/yhf/CAS(ME)II/output"

# 1. 聚合数据
data_grouped = defaultdict(list)
with open(jsonl_path, 'r') as f:
    for line in f:
        item = json.loads(line)
        key = item['image_id']
        data_grouped[key].append(item)

# 2. 构造 messages 格式 + 所有图片选取
results = []
for image_id, items in data_grouped.items():
    if not items:
        continue

    subject = items[0]['subject']
    filename = items[0]['filename']

    image_folder = os.path.join(image_root, subject, filename)
    jpg_paths = glob.glob(os.path.join(image_folder, "*.jpg"))
    png_paths = glob.glob(os.path.join(image_folder, "*.png"))

    # 合并并排序
    image_paths = sorted(jpg_paths + png_paths)

    if len(image_paths) < 1:
        print(f"[警告] 没有找到图像文件：{image_folder}")
        continue

    # 直接使用所有图片
    # 假设 image_paths 是一个包含所有图片路径的列表
    n = len(image_paths)
    selected_images = []

    if n > 1:  # 如果不止一张图片，再选前、中、后三张（排除最后一张）
        remaining_images = image_paths[:-2]  # 排除最后一张
        k = len(remaining_images)
        
        # 2. 选取前、中、后三张（如果剩余图片 >=3）
        if k >= 3:
            first = remaining_images[0]
            middle = remaining_images[k // 2]
            last_before_end = remaining_images[-1]
            selected_images.extend([first, middle, last_before_end])
        else:
            # 如果剩余图片不足3张，全部选取
            selected_images.extend(remaining_images)
    if n >= 1:
        # 1. 选取最后一张
        last_image = image_paths[-1]
        selected_images.append(last_image)


    messages = [{"role": "system", "content": '''
        You are analyzing a microexpression video sequence. 
   '''}]

    for item in items:
        messages.append({"role": "user", "content": item["question"]})
        messages.append({"role": "assistant", "content": item["answer"]})

    results.append({
        "messages": messages,
        "images": selected_images  # 所有图片被添加
    })

# 3. 保存为多模态训练格式（JSONL）
with open("converted_dataset_cas.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
