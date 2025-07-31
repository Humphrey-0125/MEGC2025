数据准备：
1. 将训练集（SAMM or CAS(ME)II) 放置到dataset数据集中
2. 依次运行data_process中的代码，完成对训练集的预处理。
3. 除此之外，你需要提前在本地配置好多媒体大语言模型（如qwen2.5-vl-7b).

训练：
1. 运行train/set_train_dataset.py构造训练集
2. 运行train.py,完成训练。

测试：
1. 运行test文件夹中的两个文件，输出测试结果。

消融实验：
1. 运行ablation/set_test_dataset.py 和 ablation/set_train_dataset.py 构造训练集和测试集，在此之前，需要你自己手工两个数据集。
2. 运行train/train.py进行训练，infer.sh完成推理
3. alation/Metrices中包含所有指标代码。