from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# 指定本地模型路径
cache_dir = "/home/d1/zwb/yhf/cache"

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    device_map="auto",
    torch_dtype="auto",
    cache_dir="/home/d1/zwb/yhf/cache/hub"
)


# 查看模型存储的本地路径
model_path = model.config._name_or_path
print(f"模型的本地存储路径：{model_path}")

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# 加载本地 processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct",local_files_only=True,cache_dir=cache_dir)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# 加载本地 processor
# processor = AutoProcessor.from_pretrained(model_dir)

# 输入消息
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "./MEGC_unseen_testset/CAS_data/dzw_a1/0000.jpg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# 构造输入
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# 推理生成
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
