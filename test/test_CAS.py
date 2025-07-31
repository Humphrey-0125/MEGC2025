import os
import json
from swift.llm import PtEngine, RequestConfig, InferRequest

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
os.environ['MAX_PIXELS'] = '1003520'
os.environ['VIDEO_MAX_PIXELS'] = '50176'
os.environ['FPS_MAX_FRAMES'] = '12'

# 设置模型路径
# model = '/home/d1/zwb/yhf/qwen2.5-vl-debug/v13-20250624-011746/checkpoint-18'
# model = '/home/d1/zwb/yhf/qwen2.5-vl-debug/v19-20250627-001049/checkpoint-51'
model = '/home/d1/zwb/yhf/qwen2.5-vl-debug/v14-20250624-212919/checkpoint-27'

# 加载推理引擎
engine = PtEngine(model, max_batch_size=2)
request_config = RequestConfig(max_tokens=512, temperature=0.5)
# 读取jsonl文件
input_jsonl_path = '/home/d1/zwb/yhf/test_dataset/me_vqa_casme3_test_to_answer.jsonl'  # 请根据实际路径修改
output_jsonl_path = 'me_vqa_casme3_test_pred.jsonl'  # 新的输出文件路径

with open(input_jsonl_path, 'r') as f:
    lines = f.readlines()

# 读取每行并处理
updated_lines = []
for line in lines:
    data = json.loads(line.strip())

    # 获取视频id和问题
    video_id = data['video_id']
    question = data['question']

    # 生成图片路径
    video_folder = f'/home/d1/zwb/yhf/MEGC_unseen_testset/ME_VQA_MEGC_2025_Test_Cropped/{data["video"]}'
    images = sorted([f for f in os.listdir(video_folder) 
                    if os.path.isfile(os.path.join(video_folder, f)) and f.endswith('.jpg')])

    # 确保视频文件夹中至少有3张图片
    if len(images) >= 3:
        # 选择第一张、最后一张和中间一张图片
        first_image = images[0]
        last_image = images[-1]
        middle_image = images[len(images) // 2]
        
        # 构建图片路径
        first_image_path = os.path.join(video_folder, first_image)
        last_image_path = os.path.join(video_folder, last_image)
        middle_image_path = os.path.join(video_folder, middle_image)
        optical_flow_image_path = os.path.join(video_folder, 'mmc_map2_nose.png')

        # 根据问题创建推理请求
        infer_request = InferRequest(messages=[{'role': 'user', 'content': 'The fourth image represents the optical flow visualization, which captures the motion dynamics between consecutive frames.In these flow maps, the colors represent the direction and magnitude of motion:Red: Right (Horizontal movement to the right).Green: Up (Vertical movement upwards). Blue: Left (Horizontal movement to the left). Yellow: Down (Vertical movement downwards)". The question is: '+question}],
                                     images=[first_image_path, middle_image_path, last_image_path,optical_flow_image_path])
        
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
