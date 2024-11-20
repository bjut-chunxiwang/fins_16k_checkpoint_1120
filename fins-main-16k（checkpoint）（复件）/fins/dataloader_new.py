import random
import numpy as np
import colorednoise
from collections import defaultdict
import torch
from torch.utils.data import Dataset
import pyloudnorm as pyln
import pandas as pd
from utils.audio import load_audio, crop_rir

class ReverbDataset(Dataset):
    """MONO RIR"""

    def __init__(self, data, config, use_noise):
        """
        Args
            data: list of tuples (file_speech, file_rir)
            config: configuration object
        """
        self.data = data  # 数据是 (file_speech, file_rir) 的元组列表
        self.config = config
        self.use_noise = use_noise

        self.rir_length = int(config.rir_duration * config.sr)  # 1 sec = 48000 samples
        self.input_signal_length = config.input_length  # 131070 samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        speech_file, rir_file = self.data[idx]

        rir = self._load_rir(rir_file)

        flipped_rir = np.flip(rir, 1).copy()
        rir = np.float32(rir)
        flipped_rir = np.float32(flipped_rir)

        source = self._load_source(speech_file)

        noise, snr_db = self._add_noise(source)

        return {
            "rir": torch.tensor(rir, dtype=torch.float32),  # 转为 torch float32
            "flipped_rir": torch.tensor(flipped_rir, dtype=torch.float32),  # 转为 torch float32
            "source": torch.tensor(source, dtype=torch.float32),  # 转为 torch float32
            "noise": torch.tensor(noise, dtype=torch.float32),  # 转为 torch float32
            "snr_db": torch.tensor(snr_db, dtype=torch.float32),  # 转为 torch float32
        }

    def _load_rir(self, rir_file):
        """Load RIR as 2D signal (channel, n_samples)"""
        rir = load_audio(rir_file, target_sr=self.config.sr)[0:1]  # 只使用一个通道
        cropped_rir = crop_rir(rir, target_length=self.rir_length)

        # TODO: normalize rir
        rir /= np.max(np.abs(rir)) * 0.999

        return np.float32(cropped_rir)  # 确保返回的 rir 是 float32

    def _load_source(self, source_file):
        source = load_audio(source_file, target_sr=self.config.sr, mono=True)[0]

        # 确保 source 是二维数组，如果是单声道的，则添加一个维度
        if source.ndim == 1:
            source = np.expand_dims(source, axis=0)

        return self._pad_or_crop(source)

    def _pad_or_crop(self, audio):
        """根据目标长度填充或裁剪音频"""
        if audio.ndim == 1:  # 如果是 1D 数组（单声道）
            audio = np.expand_dims(audio, axis=0)

        n_channels, audio_length = audio.shape
        if audio_length < self.input_signal_length:
            target_audio = np.zeros((n_channels, self.input_signal_length))
            target_audio[:, :audio_length] = audio
        else:
            target_audio = audio[:, :self.input_signal_length]
        return np.float32(target_audio)  # 确保返回的音频是 float32

    def _add_noise(self, source):
        if self.use_noise:
            if random.random() < 0.9:
                min_snr = 0.0
                max_snr = 30.0
                beta = random.random() + 1.0
                noise = colorednoise.powerlaw_psd_gaussian(beta, self.input_signal_length)
                noise = np.expand_dims(noise, 0)
                snr_db = np.array([random.random() * (max_snr - min_snr) + min_snr])
            else:
                noise = np.zeros_like(source)
                snr_db = np.array([0.0])
        else:
            noise = np.zeros_like(source)
            snr_db = np.array([0.0])

        return np.float32(noise), np.float32(snr_db)  # 确保 noise 和 snr_db 是 float32
