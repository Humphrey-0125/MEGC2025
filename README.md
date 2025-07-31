# MEGC2025 Competition Code

This repository contains the code for our solution to the MEGC2025 competition, where we achieved **third place**.

## Abstract

Micro-expression Visual Question Answering (ME-VQA) is a challenging task that combines temporal visual modeling with language-based reasoning to interpret subtle facial dynamics and emotional semantics from micro-expression videos. To address the challenges posed by short duration, weak motion signals, and limited annotations in micro-expression recognition, we propose a **temporal information-enhanced vision-language framework** designed specifically for ME-VQA.

Our method, built upon **Qwen2.5-VL-7B-Instruct**, takes both the original video sequence and a static optical flow map (computed between the onset and apex frames) as input. This dual-input setup allows the model to capture both **global temporal context** and **localized motion cues**. We employ **full-parameter supervised fine-tuning** to adapt the model for micro-expression understanding, improving its ability for fine-grained motion perception and high-level semantic reasoning. To enhance the coherence and accuracy of outputs during inference, we design **structured prompt templates**.

Experiments on the **SAMM** and **CAS(ME)\textsuperscript{3}** datasets demonstrate that our method consistently outperforms baseline models across multiple metrics, achieving an average performance improvement of **0.19**, which highlights its effectiveness and generalizability in the ME-VQA task. Our solution secured **third place** in the MEGC2025 competition. 

## System Overview

![image-20250731221328492](./assets/image-20250731221328492.png)

## Setup and Usage

### Data Preparation

1. Place the **SAMM** or **CAS(ME)II** training dataset in the `dataset/` directory.
2. Run the preprocessing scripts located in `data_process/` to prepare the training dataset.
3. In addition, ensure you have configured a multimodal large language model (such as **Qwen2.5-VL-7B**) locally.

### Training

1. Run `train/set_train_dataset.py` to construct the training dataset.
2. Execute `train.py` to start training the model.

### Testing

1. Run the two files in the `test/` folder to output the test results.

### Ablation Study

1. Run `ablation/set_test_dataset.py` and `ablation/set_train_dataset.py` to prepare the training and test datasets. Note: You will need to manually split the datasets into training and test sets beforehand.
2. Train the model by running `train/train.py`, and perform inference using `infer.sh`.
3. Evaluation metrics can be found in the `ablation/Metrics` folder.

### Results

Final results and ablation study results will be saved in the `result/` folder. The trained model is uploaded to **Google Drive**, and the aligned training dataset is available in `dataset/train_dataset.jsonl`. You can directly use the pre-trained model for testing.