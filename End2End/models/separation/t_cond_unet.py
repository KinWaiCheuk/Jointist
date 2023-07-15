import numpy as np
import librosa
import os
import sys
import math
import pickle

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchaudio
from End2End.models.separation.base import Base, init_layer, init_bn, act


from End2End.MIDI_program_map import (
                                      MIDI_Class_NUM,
                                      MIDIClassName2class_idx,
                                      class_idx2MIDIClass,
                                      )

IX_TO_LB = class_idx2MIDIClass

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation, momentum):
        super(ConvBlock, self).__init__()

        self.activation = activation

        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            dilation=(1, 1),
            padding=padding,
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            dilation=(1, 1),
            padding=padding,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x, film_dict):

        a1 = film_dict['gamma1']
        a2 = film_dict['gamma2']
        b1 = film_dict['beta1']
        b2 = film_dict['beta2']
        
        x = act(a1 * self.bn1(self.conv1(x)) + b1, self.activation)
        x = act(a2 * self.bn2(self.conv2(x)) + b2, self.activation)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, downsample, activation, momentum):
        super(EncoderBlock, self).__init__()

        self.conv_block = ConvBlock(in_channels, out_channels, kernel_size, activation, momentum)
        self.downsample = downsample

    def forward(self, x, film_dict):
        encoder = self.conv_block(x, film_dict)
        encoder_pool = F.avg_pool2d(encoder, kernel_size=self.downsample)
        return encoder_pool, encoder


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, upsample, activation, momentum):
        super(DecoderBlock, self).__init__()
        self.kernel_size = kernel_size
        self.stride = upsample
        self.activation = activation

        self.conv1 = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.stride,
            stride=self.stride,
            padding=(0, 0),
            bias=False,
            dilation=(1, 1),
        )

        self.bn1 = nn.BatchNorm2d(out_channels, momentum=momentum)

        self.conv_block2 = ConvBlock(
            out_channels * 2, out_channels, kernel_size, activation, momentum
        )

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn)

    def forward(self, input_tensor, concat_tensor, film_dict):
        x = act(self.bn1(self.conv1(input_tensor)), self.activation)
        x = torch.cat((x, concat_tensor), dim=1)
        x = self.conv_block2(x, film_dict)
        return x
    
    
class TUNetBase(nn.Module, Base):
    def __init__(self, channels_num, spec_cfg):
        super().__init__()

        n_fft = 1024
        hop_length = 320
        center = True
        pad_mode = "reflect"
        window = "hann"
        activation = "leaky_relu"
        momentum = 0.01
        is_gamma = False
        
        self.time_downsample_ratio = 2 ** 6  # This number equals 2^{#encoder_blcoks}
        self.stft = torchaudio.transforms.Spectrogram(**spec_cfg.STFT)

#         self.stft = STFT(
#             n_fft=window_size,
#             hop_length=hop_size,
#             win_length=window_size,
#             window=window,
#             center=center,
#             pad_mode=pad_mode,
#             freeze_parameters=True,
#         )

        self.istft = torchaudio.transforms.InverseSpectrogram(**spec_cfg.iSTFT)

#         self.istft = ISTFT(
#             n_fft=n_fft,
#             hop_length=hop_length,
#             win_length=n_fft,
#             window=window,
#             center=center,
#             pad_mode=pad_mode,
#             freeze_parameters=True,
#         )

        self.bn0 = nn.BatchNorm2d(n_fft // 2 + 1, momentum=momentum)

        self.encoder_block1 = EncoderBlock(
            in_channels=channels_num,
            out_channels=32,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block2 = EncoderBlock(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block3 = EncoderBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block4 = EncoderBlock(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block5 = EncoderBlock(
            in_channels=256,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block6 = EncoderBlock(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.conv_block7 = ConvBlock(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block1 = DecoderBlock(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block2 = DecoderBlock(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block3 = DecoderBlock(
            in_channels=384,
            out_channels=256,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block4 = DecoderBlock(
            in_channels=256,
            out_channels=128,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block5 = DecoderBlock(
            in_channels=128,
            out_channels=64,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block6 = DecoderBlock(
            in_channels=64,
            out_channels=32,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )

        self.after_conv_block1 = ConvBlock(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            activation=activation,
            momentum=momentum,
        )

        self.after_conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=channels_num,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )
        
        self.roll2nfft = nn.Linear(88, n_fft//2+1)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.after_conv2)

    def forward(self, input, film_dict, pianoroll):
        """
        Args:
          input: (batch_size, segment_samples, channels_num)
          isn't it (B, 1, len)?

        Outputs:
          output_dict: {
            'wav': (batch_size, segment_samples, channels_num),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """

        # input (B, 1, len)
        sp = self.stft(input)
        # (B, 1, F, T) when using torchaudio
        sp = abs(sp.transpose(-1,-2))
        """(batch_size, channels_num, time_steps, freq_bins)"""
        roll_feat = self.roll2nfft(pianoroll) # (B, F, T)
        roll_feat = roll_feat.unsqueeze(1)
        
        sp = roll_feat + sp

        # Batch normalization
        x = sp.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        """(batch_size, chanenls, time_steps, freq_bins)"""

        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]
        pad_len = (
            int(np.ceil(x.shape[2] / self.time_downsample_ratio)) * self.time_downsample_ratio
            - origin_len
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        """(batch_size, channels, padded_time_steps, freq_bins)"""

        # Let frequency bins be evenly divided by 2, e.g., 513 -> 512
        x = x[..., :-1]  # (bs, channels, T, F)

        # UNet
        (x1_pool, x1) = self.encoder_block1(x, film_dict['encoder_block1'])  # x1_pool: (bs, 32, T / 2, F / 2)
        (x2_pool, x2) = self.encoder_block2(x1_pool, film_dict['encoder_block2'])  # x2_pool: (bs, 64, T / 4, F / 4)
        (x3_pool, x3) = self.encoder_block3(x2_pool, film_dict['encoder_block3'])  # x3_pool: (bs, 128, T / 8, F / 8)
        (x4_pool, x4) = self.encoder_block4(
            x3_pool, film_dict['encoder_block4']
        )  # x4_pool: (bs, 256, T / 16, F / 16)
        (x5_pool, x5) = self.encoder_block5(
            x4_pool, film_dict['encoder_block5']
        )  # x5_pool: (bs, 384, T / 32, F / 32)
        (x6_pool, x6) = self.encoder_block6(
            x5_pool, film_dict['encoder_block6']
        )  # x6_pool: (bs, 384, T / 64, F / 64)
        x_center = self.conv_block7(x6_pool, film_dict['conv_block7'])  # (bs, 384, T / 64, F / 64)
        x7 = self.decoder_block1(x_center, x6, film_dict['decoder_block1'])  # (bs, 384, T / 32, F / 32)
        x8 = self.decoder_block2(x7, x5, film_dict['decoder_block2'])  # (bs, 384, T / 16, F / 16)
        x9 = self.decoder_block3(x8, x4, film_dict['decoder_block3'])  # (bs, 256, T / 8, F / 8)
        x10 = self.decoder_block4(x9, x3, film_dict['decoder_block4'])  # (bs, 128, T / 4, F / 4)
        x11 = self.decoder_block5(x10, x2, film_dict['decoder_block5'])  # (bs, 64, T / 2, F / 2)
        x12 = self.decoder_block6(x11, x1, film_dict['decoder_block6'])  # (bs, 32, T, F)
        x = self.after_conv_block1(x12, film_dict['after_conv_block1'])  # (bs, 32, T, F)
        x = self.after_conv2(x)  # (bs, channels, T, F)

        # Recover shape
        x = F.pad(x, pad=(0, 1))
        x = x[:, :, 0:origin_len, :]

        sp_out = torch.sigmoid(x) * sp

        # Spectrogram to wav
        length = input.shape[-1]
        wav_out = self.spectrogram_to_wav(input, sp_out, length)

        output_dict = {"waveform": wav_out}
        return output_dict    

class TUNetBase_Sum(nn.Module, Base):
    def __init__(self, channels_num, spec_cfg):
        super().__init__()

        n_fft = 1024
        hop_length = 320
        center = True
        pad_mode = "reflect"
        window = "hann"
        activation = "leaky_relu"
        momentum = 0.01
        is_gamma = False
        
        self.time_downsample_ratio = 2 ** 6  # This number equals 2^{#encoder_blcoks}
        self.stft = torchaudio.transforms.Spectrogram(**spec_cfg.STFT)

#         self.stft = STFT(
#             n_fft=window_size,
#             hop_length=hop_size,
#             win_length=window_size,
#             window=window,
#             center=center,
#             pad_mode=pad_mode,
#             freeze_parameters=True,
#         )

        self.istft = torchaudio.transforms.InverseSpectrogram(**spec_cfg.iSTFT)

#         self.istft = ISTFT(
#             n_fft=n_fft,
#             hop_length=hop_length,
#             win_length=n_fft,
#             window=window,
#             center=center,
#             pad_mode=pad_mode,
#             freeze_parameters=True,
#         )

        self.bn0 = nn.BatchNorm2d(n_fft // 2 + 1, momentum=momentum)

        self.encoder_block1 = EncoderBlock(
            in_channels=channels_num,
            out_channels=32,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block2 = EncoderBlock(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block3 = EncoderBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block4 = EncoderBlock(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block5 = EncoderBlock(
            in_channels=256,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block6 = EncoderBlock(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.conv_block7 = ConvBlock(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block1 = DecoderBlock(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block2 = DecoderBlock(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block3 = DecoderBlock(
            in_channels=384,
            out_channels=256,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block4 = DecoderBlock(
            in_channels=256,
            out_channels=128,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block5 = DecoderBlock(
            in_channels=128,
            out_channels=64,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block6 = DecoderBlock(
            in_channels=64,
            out_channels=32,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )

        self.after_conv_block1 = ConvBlock(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            activation=activation,
            momentum=momentum,
        )

        self.after_conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=channels_num,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )
        
        self.roll2nfft = nn.Linear(88, n_fft//2+1)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.after_conv2)

    def forward(self, input, film_dict, pianoroll):
        """
        Args:
          input: (batch_size, segment_samples, channels_num)
          isn't it (B, 1, len)?

        Outputs:
          output_dict: {
            'wav': (batch_size, segment_samples, channels_num),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """

        # input (B, 1, len)
        sp = self.stft(input)
        # (B, 1, F, T) when using torchaudio
        sp = abs(sp.transpose(-1,-2))
        """(batch_size, channels_num, time_steps, freq_bins)"""
        roll_feat = self.roll2nfft(pianoroll) # (B, F, T)
        roll_feat = roll_feat.unsqueeze(1)
        
        x = roll_feat + sp

        # Batch normalization
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        """(batch_size, chanenls, time_steps, freq_bins)"""

        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]
        pad_len = (
            int(np.ceil(x.shape[2] / self.time_downsample_ratio)) * self.time_downsample_ratio
            - origin_len
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        """(batch_size, channels, padded_time_steps, freq_bins)"""

        # Let frequency bins be evenly divided by 2, e.g., 513 -> 512
        x = x[..., :-1]  # (bs, channels, T, F)

        # UNet
        (x1_pool, x1) = self.encoder_block1(x, film_dict['encoder_block1'])  # x1_pool: (bs, 32, T / 2, F / 2)
        (x2_pool, x2) = self.encoder_block2(x1_pool, film_dict['encoder_block2'])  # x2_pool: (bs, 64, T / 4, F / 4)
        (x3_pool, x3) = self.encoder_block3(x2_pool, film_dict['encoder_block3'])  # x3_pool: (bs, 128, T / 8, F / 8)
        (x4_pool, x4) = self.encoder_block4(
            x3_pool, film_dict['encoder_block4']
        )  # x4_pool: (bs, 256, T / 16, F / 16)
        (x5_pool, x5) = self.encoder_block5(
            x4_pool, film_dict['encoder_block5']
        )  # x5_pool: (bs, 384, T / 32, F / 32)
        (x6_pool, x6) = self.encoder_block6(
            x5_pool, film_dict['encoder_block6']
        )  # x6_pool: (bs, 384, T / 64, F / 64)
        x_center = self.conv_block7(x6_pool, film_dict['conv_block7'])  # (bs, 384, T / 64, F / 64)
        x7 = self.decoder_block1(x_center, x6, film_dict['decoder_block1'])  # (bs, 384, T / 32, F / 32)
        x8 = self.decoder_block2(x7, x5, film_dict['decoder_block2'])  # (bs, 384, T / 16, F / 16)
        x9 = self.decoder_block3(x8, x4, film_dict['decoder_block3'])  # (bs, 256, T / 8, F / 8)
        x10 = self.decoder_block4(x9, x3, film_dict['decoder_block4'])  # (bs, 128, T / 4, F / 4)
        x11 = self.decoder_block5(x10, x2, film_dict['decoder_block5'])  # (bs, 64, T / 2, F / 2)
        x12 = self.decoder_block6(x11, x1, film_dict['decoder_block6'])  # (bs, 32, T, F)
        x = self.after_conv_block1(x12, film_dict['after_conv_block1'])  # (bs, 32, T, F)
        x = self.after_conv2(x)  # (bs, channels, T, F)

        # Recover shape
        x = F.pad(x, pad=(0, 1))
        x = x[:, :, 0:origin_len, :]

        sp_out = torch.sigmoid(x) * sp

        # Spectrogram to wav
        length = input.shape[-1]
        wav_out = self.spectrogram_to_wav(input, sp_out, length)

        output_dict = {"waveform": wav_out}
        return output_dict    
    
    
class TUNetBase_Cat(nn.Module, Base):
    def __init__(self, channels_num, spec_cfg):
        super().__init__()

        n_fft = 1024
        hop_length = 320
        center = True
        pad_mode = "reflect"
        window = "hann"
        activation = "leaky_relu"
        momentum = 0.01
        is_gamma = False
        
        self.time_downsample_ratio = 2 ** 6  # This number equals 2^{#encoder_blcoks}
        self.stft = torchaudio.transforms.Spectrogram(**spec_cfg.STFT)

#         self.stft = STFT(
#             n_fft=window_size,
#             hop_length=hop_size,
#             win_length=window_size,
#             window=window,
#             center=center,
#             pad_mode=pad_mode,
#             freeze_parameters=True,
#         )

        self.istft = torchaudio.transforms.InverseSpectrogram(**spec_cfg.iSTFT)

#         self.istft = ISTFT(
#             n_fft=n_fft,
#             hop_length=hop_length,
#             win_length=n_fft,
#             window=window,
#             center=center,
#             pad_mode=pad_mode,
#             freeze_parameters=True,
#         )

        self.bn0 = nn.BatchNorm2d(n_fft // 2 + 1, momentum=momentum)

        self.encoder_block1 = EncoderBlock(
            in_channels=channels_num,
            out_channels=32,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block2 = EncoderBlock(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block3 = EncoderBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block4 = EncoderBlock(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block5 = EncoderBlock(
            in_channels=256,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block6 = EncoderBlock(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.conv_block7 = ConvBlock(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block1 = DecoderBlock(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block2 = DecoderBlock(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block3 = DecoderBlock(
            in_channels=384,
            out_channels=256,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block4 = DecoderBlock(
            in_channels=256,
            out_channels=128,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block5 = DecoderBlock(
            in_channels=128,
            out_channels=64,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.decoder_block6 = DecoderBlock(
            in_channels=64,
            out_channels=32,
            kernel_size=(3, 3),
            upsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )

        self.after_conv_block1 = ConvBlock(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            activation=activation,
            momentum=momentum,
        )

        self.after_conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=1,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )
        
        self.roll2nfft = nn.Linear(88, n_fft//2+1)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.after_conv2)

    def forward(self, input, film_dict, pianoroll):
        """
        Args:
          input: (batch_size, segment_samples, channels_num)
          isn't it (B, 1, len)?

        Outputs:
          output_dict: {
            'wav': (batch_size, segment_samples, channels_num),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """

        # input (B, 1, len)
        sp = self.stft(input)
        # (B, 1, F, T) when using torchaudio
        sp = abs(sp.transpose(-1,-2)) # (B, 1, T, F) when using torchaudio
        """(batch_size, channels_num, time_steps, freq_bins)"""
        roll_feat = self.roll2nfft(pianoroll) # (B, T, F)
        roll_feat = roll_feat.unsqueeze(1) # (B, 1, T, F)
        sp_cat = torch.cat((roll_feat,sp),1)

        # Batch normalization
        x = sp_cat.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        """(batch_size, chanenls, time_steps, freq_bins)"""

        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]
        pad_len = (
            int(np.ceil(x.shape[2] / self.time_downsample_ratio)) * self.time_downsample_ratio
            - origin_len
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        """(batch_size, channels, padded_time_steps, freq_bins)"""

        # Let frequency bins be evenly divided by 2, e.g., 513 -> 512
        x = x[..., :-1]  # (bs, channels, T, F)

        # UNet
        (x1_pool, x1) = self.encoder_block1(x, film_dict['encoder_block1'])  # x1_pool: (bs, 32, T / 2, F / 2)
        (x2_pool, x2) = self.encoder_block2(x1_pool, film_dict['encoder_block2'])  # x2_pool: (bs, 64, T / 4, F / 4)
        (x3_pool, x3) = self.encoder_block3(x2_pool, film_dict['encoder_block3'])  # x3_pool: (bs, 128, T / 8, F / 8)
        (x4_pool, x4) = self.encoder_block4(
            x3_pool, film_dict['encoder_block4']
        )  # x4_pool: (bs, 256, T / 16, F / 16)
        (x5_pool, x5) = self.encoder_block5(
            x4_pool, film_dict['encoder_block5']
        )  # x5_pool: (bs, 384, T / 32, F / 32)
        (x6_pool, x6) = self.encoder_block6(
            x5_pool, film_dict['encoder_block6']
        )  # x6_pool: (bs, 384, T / 64, F / 64)
        x_center = self.conv_block7(x6_pool, film_dict['conv_block7'])  # (bs, 384, T / 64, F / 64)
        x7 = self.decoder_block1(x_center, x6, film_dict['decoder_block1'])  # (bs, 384, T / 32, F / 32)
        x8 = self.decoder_block2(x7, x5, film_dict['decoder_block2'])  # (bs, 384, T / 16, F / 16)
        x9 = self.decoder_block3(x8, x4, film_dict['decoder_block3'])  # (bs, 256, T / 8, F / 8)
        x10 = self.decoder_block4(x9, x3, film_dict['decoder_block4'])  # (bs, 128, T / 4, F / 4)
        x11 = self.decoder_block5(x10, x2, film_dict['decoder_block5'])  # (bs, 64, T / 2, F / 2)
        x12 = self.decoder_block6(x11, x1, film_dict['decoder_block6'])  # (bs, 32, T, F)
        x = self.after_conv_block1(x12, film_dict['after_conv_block1'])  # (bs, 32, T, F)
        x = self.after_conv2(x)  # (bs, channels, T, F)

        # Recover shape
        x = F.pad(x, pad=(0, 1))
        x = x[:, :, 0:origin_len, :]

        sp_out = torch.sigmoid(x) * sp

        # Spectrogram to wav
        length = input.shape[-1]
        
        wav_out = self.spectrogram_to_wav(input, sp_out, length)

        output_dict = {"waveform": wav_out, "roll_feat": roll_feat}
        return output_dict    

class UNetFiLM(nn.Module):
    def __init__(self, condition_size, is_gamma, is_beta=True):
        super(UNetFiLM, self).__init__()

        self.is_gamma = is_gamma
        self.is_beta = is_beta
        assert self.is_beta is True

        self.layers_num_of_conv_block = 2

        self.out_channels_dict = {
            'encoder_block1': 32, 
            'encoder_block2': 64, 
            'encoder_block3': 128, 
            'encoder_block4': 256, 
            'encoder_block5': 384, 
            'encoder_block6': 384, 
            'conv_block7': 384, 
            'decoder_block1': 384, 
            'decoder_block2': 384, 
            'decoder_block3': 256, 
            'decoder_block4': 128, 
            'decoder_block5': 64, 
            'decoder_block6': 32, 
            'after_conv_block1': 32
        }

        self.film_dict = {}

        for key in self.out_channels_dict.keys():
            self.film_dict[key] = {}

        for key in self.out_channels_dict.keys():

            out_channels = self.out_channels_dict[key]

            if self.is_gamma:
                for j in range(1, self.layers_num_of_conv_block + 1):
                    layer = nn.Linear(condition_size, out_channels, bias=True)
                    self.add_module(name='{}_gamma{}'.format(key, j), module=layer)
                    init_layer(layer)
                    self.film_dict[key]['gamma{}'.format(j)] = layer
                    
            if self.is_beta:
                for j in range(1, self.layers_num_of_conv_block + 1):
                    layer = nn.Linear(condition_size, out_channels, bias=True)
                    self.add_module(name='{}_beta{}'.format(key, j), module=layer)
                    init_layer(layer)
                    self.film_dict[key]['beta{}'.format(j)] = layer

    def forward(self, condition):

        output_dict = {}

        for key in self.out_channels_dict.keys():

            output_dict[key] = {}

            for j in range(1, self.layers_num_of_conv_block + 1):

                if self.is_gamma:
                    output_dict[key]['gamma{}'.format(j)] = self.film_dict[key]['gamma{}'.format(j)](condition)[:, :, None, None]
                else:
                    output_dict[key]['gamma{}'.format(j)] = 1.

                if self.is_beta:
                    output_dict[key]['beta{}'.format(j)] = self.film_dict[key]['beta{}'.format(j)](condition)[:, :, None, None]
                else:
                    output_dict[key]['beta{}'.format(j)] = 0.

        return output_dict


class TCondUNet(nn.Module):
    def __init__(self, mode, condition_size, is_gamma, is_beta=True, spec_cfg=None):
        super().__init__()

        self.unet_film = UNetFiLM(condition_size=condition_size, is_gamma=is_gamma, is_beta=is_beta)
        if mode=='sum':
            self.unet_base = TUNetBase_Sum(channels_num=1, spec_cfg=spec_cfg)
        elif mode=='ori':
            self.unet_base = TUNetBase(channels_num=1, spec_cfg=spec_cfg)            
        elif mode=='cat':
            self.unet_base = TUNetBase_Cat(channels_num=2, spec_cfg=spec_cfg)
        else:
            raise ValueError(f'mode={mode} is not supported')
    def forward(self, input, condition, pianoroll):

        film_dict = self.unet_film(condition)
        output_dict = self.unet_base(input, film_dict, pianoroll)

        return output_dict
