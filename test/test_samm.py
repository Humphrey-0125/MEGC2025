import os
import json
from swift.llm import PtEngine, RequestConfig, InferRequest

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
os.environ['MAX_PIXELS'] = '1003520'
os.environ['VIDEO_MAX_PIXELS'] = '50176'
os.environ['FPS_MAX_FRAMES'] = '12'

# 设置模型路径
# model = '/home/d1/zwb/yhf/qwen2.5-vl-debug/v14-20250624-212919/checkpoint-27'
# model = '/home/d1/zwb/yhf/qwen2.5-vl-debug/v13-20250624-011746/checkpoint-18'
model = '/home/d1/zwb/yhf/7_27/qwen2.5-vl-debug/v7-20250730-003056/checkpoint-5'

# 加载推理引擎
engine = PtEngine(model, max_batch_size=2)
request_config = RequestConfig(max_tokens=512, temperature=0.5)

# 读取jsonl文件
input_jsonl_path = '/home/d1/zwb/yhf/single.jsonl'  # 请根据实际路径修改
output_jsonl_path = 'me_vqa_samm_test_pred.jsonl'  # 新的输出文件路径

with open(input_jsonl_path, 'r') as f:
    lines = f.readlines()

# 读取每行并处理
updated_lines = []
for line in lines:
    data = json.loads(line.strip())

    question = data['question']

    # 生成图片路径

    # optical_flow_prompt = '''
    # The first three images represent key frames from the video. The fourth is an optical flow visualization using the HSV color model:
    # - Hue represents motion direction (red: right, green: up, blue: left, yellow: down)
    # - Saturation indicates motion consistency
    # - Value shows motion intensity
    # The question is: 
    # '''
    # 确保视频文件夹中至少有3张图片
    if 4 >= 3:

        
        # 构建图片路径
        first_image_path = "/home/d1/zwb/yhf/CAS(ME)II/output/sub01/EP02_01f/img46.jpg"
        last_image_path = "/home/d1/zwb/yhf/CAS(ME)II/output/sub01/EP02_01f/img86.jpg"
        middle_image_path = "/home/d1/zwb/yhf/CAS(ME)II/output/sub01/EP02_01f/img68.jpg"
        optical_flow_image_path = "/home/d1/zwb/yhf/CAS(ME)II/output/sub01/EP02_01f/sub01_EP02_01f_mmc_map_nose.png"
        

        # 根据问题创建推理请求
        if question == "What is the coarse expression class?":
            # 提供粗粒度情绪类别
            coarse_classes = ['negative', 'positive', 'others', 'surprise']
            infer_request = InferRequest(
                messages=[{'role': 'user', 'content': f"What is the coarse expression class?Give me answer directly. Choose from: {', '.join(coarse_classes)}"}],
                images=[first_image_path, middle_image_path, last_image_path, optical_flow_image_path]
            )
        elif question == "What is the fine-grained expression class?":
            # 提供细粒度情绪类别
            fine_grained_classes = ['anger', 'fear', 'disgust', 'sadness', 'surprise', 'pain', 'helpless', 
                                    'happiness', 'confused', 'sympathy', 'happiness']
            infer_request = InferRequest(
                messages=[{'role': 'user', 'content': f"What is the fine-grained expression class?Give me answer directly. Choose from: {', '.join(fine_grained_classes)}"}],
                images=[first_image_path, middle_image_path, last_image_path, optical_flow_image_path]
            )
        else:
            # 默认处理其他问题
            infer_request = InferRequest(
                messages=[{'role': 'user', 'content': question}],
                images=[first_image_path, middle_image_path, last_image_path]
            )
        
        # 执行推理
        resp_list = engine.infer([infer_request], request_config)

        # 更新answer字段
        answer = resp_list[0].choices[0].message.content
        data['answer'] = answer
        print(answer)
    else:
        data['answer'] = 'Not enough images in the video folder'

    updated_lines.append(json.dumps(data))

# 将更新后的内容写入到新的jsonl文件
with open(output_jsonl_path, 'w') as f:
    for updated_line in updated_lines:
        f.write(updated_line + '\n')

print(f"Answers have been updated and saved to {output_jsonl_path} successfully!")
