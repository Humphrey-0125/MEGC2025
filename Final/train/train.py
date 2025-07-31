import os
import subprocess
from transformers import modeling_utils

# Step 1: Modify ALL_PARALLEL_STYLES if it's not set
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]

print("ALL_PARALLEL_STYLES set to:", modeling_utils.ALL_PARALLEL_STYLES)

# Step 2: Set environment variables
os.environ["DISABLE_DEVICE_MAP"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "0"
os.environ["TRANSFORMERS_CACHE"] = "/home/d1/zwb/yhf/cache"
os.environ["MODELSCOPE_CACHE"] = "/home/d1/zwb/yhf/cache/modelscope"
os.environ["HF_HOME"] = "/home/d1/zwb/yhf/cache"
# Set CUDA_VISIBLE_DEVICES to use only the first 7 GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"  # Only use GPU 0 to GPU 6

# Step 3: Prepare the swift sft command for Stage 2 (Train entire model)
swift_command = [
    "swift", "sft",
    "--model", "Qwen/Qwen2.5-VL-7B-Instruct",  # Path to the checkpoint from Stage 1
    "--model_type", "qwen2_5_vl",
    "--train_type", "full",
    "--dataset", "/home/d1/zwb/yhf/base.jsonl",  # Dataset path
    "--torch_dtype", "bfloat16",
    "--freeze_vit", "false",  # Unfreeze ViT part to allow joint training
    "--freeze_llm", "false",  # Unfreeze LLM part to allow joint training
    "--freeze_aligner", "false",  # Unfreeze Aligner part to allow joint training
    "--num_train_epochs", "2",  # Increase epochs for more fine-tuning
    "--per_device_train_batch_size", "1",  # Adjust batch size as per memory constraints
    "--gradient_accumulation_steps", "8",  # Increase gradient accumulation for stable training
    "--learning_rate", "5e-6",  # Lower learning rate for finer updates
    "--eval_steps", "-1",  # Disable evaluation during training, set to a number for periodic evaluation
    "--save_steps", "1000",  # Save the model every 1000 steps
    "--save_total_limit", "10",  # Limit the number of saved models to the latest 10
    "--logging_steps", "5",  # Log every 5 steps
    "--max_length", "10000",  # Max sequence length
    "--output_dir", "/home/d1/zwb/yhf/7_30/qwen2.5-vl-debug",  # Output directory for saving the model
    "--warmup_ratio", "0.05",  # Warm-up ratio for learning rate
    "--dataloader_num_workers", "0",  # Number of workers for loading data
    "--dataset_num_proc", "7",  # Number of processes for dataset processing
    # "--deepspeed", "zero2"  # DeepSpeed optimizer for efficient training
]

# Step 4: Execute the command for Stage 2 training
subprocess.run(swift_command)
