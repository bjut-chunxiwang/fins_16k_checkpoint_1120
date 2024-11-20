import os
import librosa
import numpy as np
from sklearn.metrics import mean_squared_error

# 设置文件夹路径
folder_path = '/home/cxw/GAN/fins-main-16k/fins/saved_data/audio'

# 获取所有音频文件
files = os.listdir(folder_path)

# 筛选出predicted_rir和true_rir的文件
predicted_files = sorted([f for f in files if f.startswith('predicted_rir') and f.endswith('.wav')])
true_files = sorted([f for f in files if f.startswith('true_rir') and f.endswith('.wav')])

# 初始化MSE差异列表
mse_differences = []

# 遍历每对文件计算MSE
for pred_file, true_file in zip(predicted_files, true_files):
    # 加载音频文件
    pred_audio, sr = librosa.load(os.path.join(folder_path, pred_file), sr=None)
    true_audio, sr = librosa.load(os.path.join(folder_path, true_file), sr=None)

    # 确保两个音频文件长度相同，若不同可以进行裁剪
    min_len = min(len(pred_audio), len(true_audio))
    pred_audio = pred_audio[:min_len]
    true_audio = true_audio[:min_len]

    # 计算MSE
    mse = mean_squared_error(true_audio, pred_audio)
    mse_differences.append(mse)

# 输出所有MSE差异
for i, mse in enumerate(mse_differences):
    print(f"第{i + 1}对文件的MSE差异: {mse}")

# 计算并输出平均MSE差异
average_mse = np.mean(mse_differences)
print(f"\n所有MSE差异的平均值: {average_mse}")
