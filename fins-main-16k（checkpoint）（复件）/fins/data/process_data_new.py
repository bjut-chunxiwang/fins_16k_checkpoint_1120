import pandas as pd
import random
from pathlib import Path

random.seed(0)

def load_dataset(csv_path):
    # 加载 CSV 文件并提取列
    df = pd.read_csv(csv_path)
    file_speech = df['file_speech'].tolist()
    file_rir = df['file_rir'].tolist()
    split_labels = df['split'].tolist()
    return file_speech, file_rir, split_labels

def split_data(file_speech, file_rir, split_labels):
    train_files, valid_files, test_files = [], [], []
    for speech, rir, split in zip(file_speech, file_rir, split_labels):
        if split == 'train':
            train_files.append((speech, rir))
        elif split == 'val':
            valid_files.append((speech, rir))
        elif split == 'test':
            test_files.append((speech, rir))
    return train_files, valid_files, test_files

def load_rir_dataset(csv_path):
    file_speech, file_rir, split_labels = load_dataset(csv_path)
    return split_data(file_speech, file_rir, split_labels)

def load_speech_dataset(csv_path):
    # 返回语音数据集与 RIR 数据集在同一函数中处理
    return load_rir_dataset(csv_path)


if __name__ == "__main__":
    csv_path = "/home/cxw/GAN/fins-main/fins/read.csv"  # 替换为实际路径
    train_data, valid_data, test_data = load_rir_dataset(csv_path)
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(valid_data)}")
    print(f"Test samples: {len(test_data)}")