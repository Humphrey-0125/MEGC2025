import json

input_path = "/home/d1/zwb/yhf/converted_dataset.jsonl"
output_path = "train_dataset_prompt.jsonl"

with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
    for line in infile:
        data = json.loads(line)
        system_msg = None
        question_answer_pairs = []
        new_images = data.get("images", [])

        # 拆分 messages 中的内容
        messages = data["messages"]
        for i in range(len(messages)):
            if messages[i]["role"] == "system":
                system_msg = messages[i]
            elif messages[i]["role"] == "user":
                user_msg = messages[i]
                # 假设后面紧跟的是 assistant 回答
                if i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
                    assistant_msg = messages[i + 1]
                    question_answer_pairs.append((user_msg, assistant_msg))

        # 为每一个问答对生成一个单独的样本
        for user_msg, assistant_msg in question_answer_pairs:
            new_entry = {
                "messages": [system_msg, user_msg, assistant_msg],
                "images": new_images
            }
            outfile.write(json.dumps(new_entry) + "\n")

print("转换完成，结果保存在:", output_path)
