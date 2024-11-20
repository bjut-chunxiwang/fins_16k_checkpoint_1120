import os
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt

from loss import MultiResolutionSTFTLoss
from utils.audio import batch_convolution, add_noise_batch, audio_normalize_batch

torch.autograd.set_detect_anomaly(True)


class Trainer:
    def __init__(self, model, train_data, valid_data, config, eval_config, args):
        self.model_name = f"{args.save_name}-{datetime.now().strftime('%y%m%d-%H%M%S')}"
        self.train_data = train_data
        self.valid_data = valid_data

        self.device = args.device
        self.config = config
        self.eval_config = eval_config
        self.args = args
        self.model = model
        # 初始化开始的epoch
        self.start_epoch = 0
        # 初始化模型
        self._init_model()

        self.model_checkpoint_dir = os.path.join(config.checkpoint_dir, self.model_name)
        self.disc_checkpoint_dir = os.path.join(config.checkpoint_dir, self.model_name + '-disc')

        self.writer = SummaryWriter(os.path.join(config.logging_dir, self.model_name))
        if not os.path.exists(self.model_checkpoint_dir):
            os.makedirs(self.model_checkpoint_dir, exist_ok=True)

        if not os.path.exists(self.disc_checkpoint_dir):
            os.makedirs(self.disc_checkpoint_dir, exist_ok=True)

    def _init_model(self):
        self.model = self.model.to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total params: {total_params}")

        # Loss
        fft_sizes = [64, 512, 2048, 8192]
        hop_sizes = [32, 256, 1024, 4096]
        win_lengths = [64, 512, 2048, 8192]
        sc_weight = 1.0
        mag_weight = 1.0

        self.stft_loss_fn = MultiResolutionSTFTLoss(
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_lengths=win_lengths,
            sc_weight=sc_weight,
            mag_weight=mag_weight,
        ).to(self.device)

        self.recon_stft_loss_fn = MultiResolutionSTFTLoss(
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_lengths=win_lengths,
            sc_weight=sc_weight,
            mag_weight=mag_weight,
        ).to(self.device)

        self.loss_dict = {}

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=1e-6)

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.lr_step_size,
            gamma=self.config.lr_decay_factor,
        )

        # 如果指定了检查点路径，则加载检查点
        if self.args.checkpoint_path:
            state_dicts = torch.load(self.args.checkpoint_path, map_location=self.device)

            self.model.load_state_dict(state_dicts["model_state_dict"])

            if "optim_state_dict" in state_dicts.keys():
                self.optimizer.load_state_dict(state_dicts["optim_state_dict"])

            if "sched_state_dict" in state_dicts.keys():
                self.scheduler.load_state_dict(state_dicts["sched_state_dict"])

            # 恢复开始的epoch
            if "epoch" in state_dicts.keys():
                self.start_epoch = state_dicts["epoch"] + 1
            else:
                self.start_epoch = 0

    def make_batch_data(self, batch):
        flipped_rir = batch["flipped_rir"].to(self.device)
        source = batch['source'].to(self.device)

        reverberated_source = batch_convolution(source, flipped_rir)

        noise = batch['noise'].to(self.device)
        snr_db = batch['snr_db'].to(self.device)

        batch_size, _, _ = noise.size()

        reverberated_source = audio_normalize_batch(reverberated_source, "rms", self.config.rms_level)

        # 添加噪声
        reverberated_source_with_noise = add_noise_batch(reverberated_source, noise, snr_db)

        # 随机噪声用于后期部分
        rir_length = int(self.config.rir_duration * self.config.sr)
        stochastic_noise = torch.randn((batch_size, 1, rir_length), device=self.device)
        batch_stochastic_noise = stochastic_noise.repeat(1, self.config.num_filters, 1)

        # 解码器条件的噪声
        batch_noise_condition = torch.randn((batch_size, self.config.noise_condition_length), device=self.device)

        return (
            reverberated_source_with_noise,
            reverberated_source,
            batch_stochastic_noise,
            batch_noise_condition,
        )

    def train(self):
        for epoch in range(self.start_epoch, self.config.num_epochs):
            self.model.train()

            torch.cuda.empty_cache()
            for i, batch in enumerate(self.train_data):
                rir = batch['rir'].to(self.device)

                # 准备批量数据
                (
                    reverberated_source_with_noise,
                    reverberated_source,
                    batch_stochastic_noise,
                    batch_noise_condition,
                ) = self.make_batch_data(batch)

                # 前向传播
                predicted_rir = self.model(
                    reverberated_source_with_noise, batch_stochastic_noise, batch_noise_condition
                )

                total_loss = 0.0

                # 计算损失
                stft_loss_dict = self.stft_loss_fn(predicted_rir, rir)
                stft_loss = stft_loss_dict["total"]
                sc_loss = stft_loss_dict["sc_loss"].item()
                mag_loss = stft_loss_dict["mag_loss"].item()

                total_loss = total_loss + stft_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_value)

                self.optimizer.step()

                if i % 1 == 0:
                    print(
                        "epoch",
                        epoch,
                        "batch",
                        i,
                        "total loss",
                        stft_loss.item(),
                    )

            # 验证
            if (epoch + 1) % self.config.validation_interval == 0:
                print("Validating...")
                self.model.eval()

                with torch.no_grad():
                    valid_loss = self.validate()
                    print(f"Validation loss : {valid_loss}")

                    self.writer.add_scalar(f"total/valid", valid_loss, global_step=epoch)

                    self.writer.flush()

                self.model.train()

            self.scheduler.step()

            # 日志记录
            print(self.model_name)
            print(
                f"Train {epoch}/{self.config.num_epochs} - loss: {total_loss.item():.3f}, stft_loss: {stft_loss.item():.3f}, sc_loss: {sc_loss:.3f}, mag_loss: {mag_loss:.3f}"
            )
            print(f"Curr lr : {self.scheduler.get_last_lr()}")

            self.writer.add_scalar("sc_loss/train", sc_loss, global_step=epoch)
            self.writer.add_scalar("mag_loss/train", mag_loss, global_step=epoch)
            self.writer.add_scalar("loss/train", total_loss.item(), global_step=epoch)

            self.writer.flush()

            # 绘图

            if (epoch + 1) % self.config.random_plot_interval == 0:
                print("Plotting at epoch", epoch)
                self.model.eval()
                with torch.no_grad():
                    for nth_batch, batch in enumerate(self.valid_data):
                        print("nth batch", nth_batch)
                        self.plot(batch, nth_batch, epoch)
                        break

                self.model.train()

            # 保存模型
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                print("Saving model at epoch", epoch)
                # 保存模型
                state_dicts = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optim_state_dict": self.optimizer.state_dict(),
                    "sched_state_dict": self.scheduler.state_dict(),
                }

                torch.save(state_dicts, os.path.join(self.model_checkpoint_dir, f"epoch-{epoch}.pt"))

    def validate(self):
        total_loss = 0.0
        for i, batch in enumerate(self.valid_data):
            rir = batch['rir'].to(self.device)

            # 准备批量数据
            (
                reverberated_source_with_noise,
                reverberated_source,
                batch_stochastic_noise,
                batch_noise_condition,
            ) = self.make_batch_data(batch)

            # 前向传播
            predicted_rir = self.model(
                reverberated_source_with_noise, batch_stochastic_noise, batch_noise_condition
            )

            # 计算损失
            stft_loss_dict = self.stft_loss_fn(predicted_rir, rir)
            stft_loss = stft_loss_dict["total"].item()

            total_loss = total_loss + stft_loss

        n_valid_data = len(self.valid_data)
        return total_loss / n_valid_data

    def plot(self, batch, nth_batch, epoch):
        print("Plotting...")
        # 准备批量数据
        (
            total_reverberated_source_with_noise,
            total_reverberated_source,
            batch_stochastic_noise,
            batch_noise_condition,
        ) = self.make_batch_data(batch)

        # 前向传播
        predicted_rir = self.model(total_reverberated_source_with_noise, batch_stochastic_noise, batch_noise_condition)

        rir = batch['rir'].to(self.device)
        source = batch['source'].to(self.device)

        flip_predicted_rir = torch.flip(predicted_rir, dims=[2])

        reverberated_speech_predicted = batch_convolution(source, flip_predicted_rir)
        reverberated_speech_predicted = audio_normalize_batch(
            reverberated_speech_predicted, "rms", self.config.rms_level
        )

        import os
        from scipy.io import wavfile
        import numpy as np

        # 保存文件夹
        self.config.save_folder = 'saved_data'

        # 确保保存文件夹存在
        os.makedirs(self.config.save_folder, exist_ok=True)

        for i in range(self.config.batch_size):
            if i >= 10:  # 限制只保存前10个文件
                break
            curr_true_rir = rir[i, 0]
            curr_predicted_rir = predicted_rir[i, 0]
            plt.figure()
            plt.ylim([-self.config.peak_norm_value, self.config.peak_norm_value])
            plt.plot(curr_true_rir.cpu().numpy()[:10000], label='True RIR')
            plt.plot(curr_predicted_rir.cpu().numpy()[:10000], label='Predicted RIR')
            plt.legend()
            # 保存图像到指定文件夹
            plot_save_path = os.path.join(self.config.save_folder,
                                          f"rir_plot_{nth_batch * self.config.batch_size + i}.png")
            plt.savefig(plot_save_path)
            # 添加到TensorBoard
            self.writer.add_figure(f"rir/{nth_batch * self.config.batch_size + i}", plt.gcf(), global_step=epoch)
            plt.close()

            # 保存音频到指定文件夹
            audio_save_folder = os.path.join(self.config.save_folder, 'audio')
            os.makedirs(audio_save_folder, exist_ok=True)

            # 设置采样率（根据实际情况设置，假设为16000 Hz）
            sample_rate = 16000

            # 将Tensor转换为NumPy数组并确保数据类型为float32
            true_rir_audio = curr_true_rir.cpu().numpy().astype(np.float32)
            predicted_rir_audio = curr_predicted_rir.cpu().numpy().astype(np.float32)

            # 归一化音频数据到[-1, 1]范围内（如果需要）
            max_val = max(np.abs(true_rir_audio).max(), np.abs(predicted_rir_audio).max(), 1e-8)
            true_rir_audio_normalized = true_rir_audio / max_val
            predicted_rir_audio_normalized = predicted_rir_audio / max_val

            # 保存真实RIR音频
            true_audio_save_path = os.path.join(audio_save_folder,
                                                f"true_rir_{nth_batch * self.config.batch_size + i}.wav")
            wavfile.write(true_audio_save_path, sample_rate, true_rir_audio_normalized)

            # 保存预测RIR音频
            predicted_audio_save_path = os.path.join(audio_save_folder,
                                                     f"predicted_rir_{nth_batch * self.config.batch_size + i}.wav")
            wavfile.write(predicted_audio_save_path, sample_rate, predicted_rir_audio_normalized)

        # 刷新TensorBoard
        self.writer.flush()
