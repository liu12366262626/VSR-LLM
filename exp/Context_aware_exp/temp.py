import torch

# 指定模型文件路径
model_path = '/work/liuzehua/task/VSR/VSP-LLM/main_log/2024-08-27/22-35-51/model_v3_vsp-llm-90h-single/checkpoints/checkpoint_8_7500.pt'

# 加载模型
model = torch.load(model_path)
temp = []
for key,value in model['model'].items():
    temp.append((key,value))

# 如果你仅仅保存了模型的状态字典（state_dict），你需要先定义模型结构，然后加载状态字典
# model = YourModelClass()  # 替换为你的模型类
# model.load_state_dict(torch.load(model_path))

# 如果要把模型放到GPU上
model = model.to('cuda')  # 如果你有GPU并且希望在GPU上运行

# 现在你可以使用这个模型来做推理或继续训练
