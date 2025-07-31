
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model /home/d1/zwb/yhf/7_30/qwen2.5-vl-debug/v0-20250730-011134/checkpoint-570 \
    --stream true \
    --infer_backend pt \
    --val_dataset /home/d1/zwb/yhf/test_split_unlabeled.jsonl \
    --max_new_tokens 2048 \
    --temperature 0.5 \
    --top_p 0.5 
