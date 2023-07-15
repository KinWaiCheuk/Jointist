import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank

from jointist.models.utils import init_layer, init_bn
from jointist.models.transcription_models import AcousticModelCRnn8Dropout
from jointist.config import SAMPLE_RATE, FRAMES_PER_SECOND, BN_MOMENTUM

import sys


def get_model_class(model_type):
    r"""Get model.

    Args:
        model_type: str, e.g., 'CRNN'

    Returns:
        nn.Module
    """
    if model_type == 'Cnn14':
        return Cnn14
    
    if model_type == 'KinWaiCnn14':
        return KinWaiCnn14

    elif model_type == 'CRNN':
        return CRNN

    elif model_type == 'Cnn14cond':
        return Cnn14cond

    else:
        raise NotImplementedError


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


class Cnn14(nn.Module):
    def __init__(self, classes_num):

        super(Cnn14, self).__init__()

        sample_rate = 16000
        window_size = 1024
        hop_size = 160
        mel_bins = 88
        fmin = 0
        fmax = 8000

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True,
        )

        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.conv_block1 = ConvBlock(in_channels=2, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, waveform, onset_roll):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(waveform)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = torch.cat((x, onset_roll[:, None, :, :]), dim=1)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict
    
    
class Cnn14MeanMax(nn.Module):
    def __init__(self, classes_num):

        super().__init__()

        sample_rate = 16000
        window_size = 1024
        hop_size = 160
        mel_bins = 229
        fmin = 0
        fmax = 8000

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True,
        )

        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
        self.fc2 = nn.Linear(2048, 2048, bias=True)
        self.fc_audiocount = nn.Linear(2048, 1, bias=True)        

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, waveform):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(waveform)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)
        spec = x
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training) # (8, 2048, 31, 7)
        x = torch.mean(x, dim=3) # (8, 2048, 31)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        embedding = x1 + x2

        x = F.dropout(embedding, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
#         embedding_class = F.dropout(x, p=0.5, training=self.training)
        instrument_class = torch.sigmoid(self.fc_audioset(x))
        
        x = F.dropout(embedding, p=0.5, training=self.training)
        x = F.relu_(self.fc2(x))
#         embedding = F.dropout(x, p=0.5, training=self.training)
        instrument_count = self.fc_audiocount(x)
    
        output_dict = {'instrument_class': instrument_class, 'instrument_count': instrument_count, 'spec': spec}

        return output_dict    

class Cnn14Seq2Seq_AutoRegressive(nn.Module):
    def __init__(self, classes_num):

        super().__init__()

        sample_rate = 16000
        window_size = 1024
        hop_size = 160
        mel_bins = 229
        fmin = 0
        fmax = 8000

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        
        self.classes_num = classes_num

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True,
        )

        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        
        self.lstm_enc = nn.LSTM(input_size=2048, hidden_size=2048, batch_first=True, bidirectional=False)
        
        self.embedding = nn.Embedding(classes_num+2, 2048)        
        self.lstm_dec = nn.LSTM(input_size=2048, hidden_size=2048, batch_first=True, bidirectional=False)
        self.linear_dec = nn.Linear(2048, classes_num+1, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
#         init_layer(self.fc1)
#         init_layer(self.fc_audioset)

    def forward(self, waveform):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(waveform)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)
        spec = x
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training) # (8, 2048, 31, 7)
        x = torch.mean(x, dim=3) # (8, 2048, 31)
        x = x.transpose(1,2) # (8, 31, 2048)

        _, h = self.lstm_enc(x)
              
#       SOS_token = 0
#       EOS_token = 1     
        self.SOS_token = self.classes_num*torch.ones(x.shape[0]).long().to(x.device)
        decoder_input = self.SOS_token.to(x.device)
        decoder_hidden = h
        decoder_outputs = torch.ones(x.shape[0], x.shape[1], self.classes_num+1).to(x.device)
        for di in range(x.shape[1]): # TODO: what is the maxium num_instruments?
            decoder_input = self.embedding(decoder_input).unsqueeze(1) # (B,1, embedding)
            decoder_output, decoder_hidden = self.lstm_dec(decoder_input, decoder_hidden)
            decoder_output = self.linear_dec(decoder_output)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            decoder_outputs[:,di] = decoder_output.squeeze(1)

        output_dict = {'decoder_outputs': decoder_outputs, 'spec': spec}

        return output_dict
    
    
class Cnn14Seq2Seq_biLSTM(nn.Module):
    def __init__(self, classes_num):

        super().__init__()

        sample_rate = 16000
        window_size = 1024
        hop_size = 160
        mel_bins = 229
        fmin = 0
        fmax = 8000

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        
        self.classes_num = classes_num

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True,
        )

        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        
        self.lstm_enc = nn.LSTM(input_size=2048, hidden_size=2048, batch_first=True, bidirectional=True)      
        self.lstm_dec = nn.LSTM(input_size=4096, hidden_size=2048, batch_first=True, bidirectional=True)
        self.linear_dec = nn.Linear(4096, classes_num+1, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
#         init_layer(self.fc1)
#         init_layer(self.fc_audioset)

    def forward(self, waveform):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(waveform)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)
        spec = x
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training) # (8, 2048, 31, 7)
#         x = torch.mean(x, dim=3) # (8, 2048, 31)
        x = x.flatten(2) # (8, 2048, 31*7)
        x = x.transpose(1,2) # (8, 31, 2048)
        
        x, h = self.lstm_enc(x)
        x, h = self.lstm_dec(x, h)        
        
        decoder_output = self.linear_dec(x)

        output_dict = {'decoder_outputs': decoder_output, 'spec': spec}

        return output_dict
    
    
class Cnn14Seq2Seq_LSTM(nn.Module):
    def __init__(self, classes_num):

        super().__init__()

        sample_rate = 16000
        window_size = 1024
        hop_size = 160
        mel_bins = 229
        fmin = 0
        fmax = 8000

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        
        self.classes_num = classes_num

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True,
        )

        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        
        self.lstm_enc = nn.LSTM(input_size=2048, hidden_size=2048, batch_first=True, bidirectional=False)      
        self.lstm_dec = nn.LSTM(input_size=2048, hidden_size=2048, batch_first=True, bidirectional=False)
        self.linear_dec = nn.Linear(2048, classes_num+1, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
#         init_layer(self.fc1)
#         init_layer(self.fc_audioset)

    def forward(self, waveform):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(waveform)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)
        spec = x
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training) # (8, 2048, 31, 7)
        x = torch.mean(x, dim=3) # (8, 2048, 31)
        x = x.transpose(1,2) # (8, 31, 2048)

        x, h = self.lstm_enc(x)
        x, h = self.lstm_dec(x, h)        
        
        decoder_output = self.linear_dec(x)

        output_dict = {'decoder_outputs': decoder_output, 'spec': spec}

        return output_dict     
    
class Cnn14LSTM(nn.Module):
    def __init__(self, classes_num):

        super().__init__()

        sample_rate = 16000
        window_size = 1024
        hop_size = 160
        mel_bins = 229
        fmin = 0
        fmax = 8000

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True,
        )

        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        
        self.lstm = nn.LSTM(input_size=2048, hidden_size=2048, batch_first=True, bidirectional=True)
        
        self.fc1 = nn.Linear(4096, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
        self.fc2 = nn.Linear(4096, 2048, bias=True)
        self.fc_audiocount = nn.Linear(2048, 1, bias=True)        

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, waveform):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(waveform)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)
        spec = x
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        x, h = self.lstm(x.transpose(1,2))
        embedding = h[0].transpose(0,1).flatten(1) # (B, dim*2)    
        

        x = F.dropout(embedding, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
#         embedding_class = F.dropout(x, p=0.5, training=self.training)
        instrument_class = torch.sigmoid(self.fc_audioset(x))
        
        x = F.dropout(embedding, p=0.5, training=self.training)
        x = F.relu_(self.fc2(x))
#         embedding = F.dropout(x, p=0.5, training=self.training)
        instrument_count = self.fc_audiocount(x)        
    
        output_dict = {'instrument_class': instrument_class, 'instrument_count': instrument_count, 'spec': spec}

        return output_dict    


class CRNN(nn.Module):
    def __init__(self, classes_num):
        super(CRNN, self).__init__()

        sample_rate = 16000
        window_size = 1024
        hop_size = 160
        mel_bins = 229
        fmin = 0
        fmax = 8000

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        midfeat = 640

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True,
        )

        self.bn0 = nn.BatchNorm2d(mel_bins, momentum=BN_MOMENTUM)

        self.acoustic_model = AcousticModelCRnn8Dropout(512, midfeat, in_channels=2)
        
        self.fc_final = nn.Linear(512, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc_final)

    def forward(self, waveform, onset_roll):
        r"""
        Args:
          input: (batch_size, data_length)

        Outputs:
          output_dict: dict, {
            'reg_onset_output': (batch_size, time_steps, classes_num),
            'reg_offset_output': (batch_size, time_steps, classes_num),
            'frame_output': (batch_size, time_steps, classes_num)
          }
        """
        x = self.spectrogram_extractor(waveform)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = torch.cat((x, onset_roll[:, None, :, :]), dim=1)

        x = self.acoustic_model(x)  # (batch_size, time_steps, classes_num)

        frames_num = x.shape[1]
        embedding = x[:, frames_num // 2, :]

        clipwise_output = torch.sigmoid(self.fc_final(embedding))

        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict

        return output_dict


##########
class ConvBlockCond(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlockCond, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.beta1 = nn.Linear(88, out_channels)
        self.beta2 = nn.Linear(88, out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, onset_roll, pool_size=(2, 2), pool_type='avg'):
        """
        input: (batch_size, 1, time_steps, pitch_notes)
        onset_roll, (batch_size, time_steps, pitch_notes)
        """
        b1 = self.beta1(onset_roll).transpose(1, 2)[:, :, :, None]
        b2 = self.beta2(onset_roll).transpose(1, 2)[:, :, :, None]

        x = input
        x = F.relu_(self.bn1(self.conv1(x)) + b1)
        x = F.relu_(self.bn2(self.conv2(x)) + b2)
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


class Cnn14cond(nn.Module):
    def __init__(self, classes_num):

        super(Cnn14cond, self).__init__()

        sample_rate = 16000
        window_size = 1024
        hop_size = 160
        mel_bins = 229
        fmin = 0
        fmax = 8000

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True,
        )

        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.conv_block1 = ConvBlockCond(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlockCond(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlockCond(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlockCond(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlockCond(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlockCond(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, waveform, onset_roll):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(waveform)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        c = onset_roll
        x = self.conv_block1(x, c, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        c = F.avg_pool1d(c.transpose(1, 2), kernel_size=2).transpose(1, 2)

        x = self.conv_block2(x, c, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        c = F.avg_pool1d(c.transpose(1, 2), kernel_size=2).transpose(1, 2)

        x = self.conv_block3(x, c, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        c = F.avg_pool1d(c.transpose(1, 2), kernel_size=2).transpose(1, 2)

        x = self.conv_block4(x, c, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        c = F.avg_pool1d(c.transpose(1, 2), kernel_size=2).transpose(1, 2)

        x = self.conv_block5(x, c, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        c = F.avg_pool1d(c.transpose(1, 2), kernel_size=2).transpose(1, 2)

        x = self.conv_block6(x, c, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict