import json

input_path = "/home/d1/zwb/yhf/test_dataset_no.jsonl"
output_path = "test_split_unlabeled_no_prompt.jsonl"

with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
    for line in infile:
        data = json.loads(line)
        messages = data["messages"]
        images = data.get("images", [])

        system_msg = None
        user_msgs = []

        # 拆解 messages
        for i in range(len(messages)):
            if messages[i]["role"] == "system":
                system_msg = messages[i]
            elif messages[i]["role"] == "user":
                user_msgs.append(messages[i])

        # 输出每个问题作为一条新记录
        for user_msg in user_msgs:
            new_entry = {
                "messages": [system_msg, user_msg],
                "images": images
            }
            outfile.write(json.dumps(new_entry) + "\n")

print("✅ 无答案版本生成完成，输出路径:", output_path)
