import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import transforms
from torchaudio.functional import amplitude_to_DB

from End2End.models.utils import init_bn, init_gru, init_layer
from End2End.constants import BN_MOMENTUM, SAMPLE_RATE

epsilon=1e-10


def get_model_class(model_type):
    r"""Get model.

    Args:
        model_type: str, e.g., 'CRNN'

    Returns:
        nn.Module
    """
    if model_type == 'CRNN':
        return CRNN

    else:
        raise NotImplementedError()


def init_embedding(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.uniform_(layer.weight, -1., 1.)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, condition_size):

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

        self.bn1 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)

        self.beta1 = nn.Linear(condition_size, out_channels, bias=True)
        self.beta2 = nn.Linear(condition_size, out_channels, bias=True)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        init_embedding(self.beta1)
        init_embedding(self.beta2)

    def forward(self, input, condition, pool_size=(2, 2), pool_type='avg'):
        r"""
        Args:
          input: (batch_size, in_channels, time_steps, freq_bins)

        Outputs:
          output: (batch_size, out_channels, classes_num)
        """

        b1 = self.beta1(condition)[:, :, None, None]
        b2 = self.beta2(condition)[:, :, None, None]

        x = F.relu_(self.bn1(self.conv1(input)) + b1)
        x = F.relu_(self.bn2(self.conv2(x)) + b2)

        if pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)

        return x
    
    
class CNN8Dropout(nn.Module):
    def __init__(self, condition_size, in_channels=1):
        super().__init__()
        out_channels = 128
        
        self.conv_block1 = ConvBlock(in_channels=in_channels, out_channels=48, condition_size=condition_size)
        self.conv_block2 = ConvBlock(in_channels=48, out_channels=64, condition_size=condition_size)
        self.conv_block3 = ConvBlock(in_channels=64, out_channels=96, condition_size=condition_size)
        self.conv_block4 = ConvBlock(in_channels=96, out_channels=out_channels, condition_size=condition_size)
        
        self.freq_dim = 14
        self.out_channels = 128
        
    def forward(self, input, condition):
        x = self.conv_block1(input, condition, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, condition, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, condition, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, condition, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)  # (batch, ch, time, freq)
        return x
    
    
class CNN8Dropout_Wide(nn.Module):
    def __init__(self, condition_size, in_channels=1):
        super().__init__()
        out_channels = 512
        
        self.conv_block1 = ConvBlock(in_channels=in_channels, out_channels=64, condition_size=condition_size)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128, condition_size=condition_size)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256, condition_size=condition_size)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=out_channels, condition_size=condition_size)
        
        self.freq_dim = 14
        self.out_channels = out_channels
        
    def forward(self, input, condition):
        x = self.conv_block1(input, condition, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, condition, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, condition, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, condition, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)  # (batch, ch, time, freq)
        return x    
        
    

class CNN14Dropout(nn.Module):
    def __init__(self, condition_size, in_channels=1):
        super().__init__()
        out_channels = 484
        
        self.conv_block1 = ConvBlock(in_channels=in_channels, out_channels=48, condition_size=condition_size)
        self.conv_block2 = ConvBlock(in_channels=48, out_channels=64, condition_size=condition_size)
        self.conv_block3 = ConvBlock(in_channels=64, out_channels=96, condition_size=condition_size)
        self.conv_block4 = ConvBlock(in_channels=96, out_channels=160, condition_size=condition_size)
        self.conv_block5 = ConvBlock(in_channels=160, out_channels=228, condition_size=condition_size)        
        self.conv_block6 = ConvBlock(in_channels=228, out_channels=out_channels, condition_size=condition_size)
        
        
        self.freq_dim = 3
        self.out_channels = out_channels
        
    def forward(self, input, condition):
        x = self.conv_block1(input, condition, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, condition, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, condition, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, condition, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)  # (batch, ch, time, freq)
        x = self.conv_block5(x, condition, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)  # (batch, ch, time, freq)
        x = self.conv_block6(x, condition, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)  # (batch, ch, time, freq)
        return x
    
    
class CNN14Dropout_Wide(nn.Module):
    def __init__(self, condition_size, in_channels=1):
        super().__init__()
        out_channels = 2048
        
        self.conv_block1 = ConvBlock(in_channels=in_channels, out_channels=64, condition_size=condition_size)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128, condition_size=condition_size)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256, condition_size=condition_size)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512, condition_size=condition_size)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024, condition_size=condition_size)        
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=out_channels, condition_size=condition_size)
        
        
        self.freq_dim = 3
        self.out_channels = out_channels
        
    def forward(self, input, condition):
        x = self.conv_block1(input, condition, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, condition, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, condition, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, condition, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)  # (batch, ch, time, freq)
        x = self.conv_block5(x, condition, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)  # (batch, ch, time, freq)
        x = self.conv_block6(x, condition, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)  # (batch, ch, time, freq)
        return x
    



class Original(nn.Module):
    def __init__(self, spec_args, frames_per_second, condition_size, classes_num, modeling_offset, modeling_velocity=None):
        super().__init__()

        self.frames_per_second = frames_per_second
        self.classes_num = classes_num
        
        self.spec_layer = transforms.MelSpectrogram(**spec_args.STFT)

        midfeat = 3584

        self.bn0 = nn.BatchNorm2d(spec_args.STFT.n_mels, momentum=BN_MOMENTUM)

        self.frame_model = AcousticModelCRnn8Dropout(condition_size, classes_num, midfeat)
        self.reg_onset_model = AcousticModelCRnn8Dropout(condition_size, classes_num, midfeat)

        if modeling_offset:
            self.reg_offset_model = AcousticModelCRnn8Dropout(condition_size, classes_num, midfeat)
            self.final_gru_ratio = 3
        else:
            self.reg_offset_model = None
            self.final_gru_ratio = 2

        if modeling_velocity:
            self.velocity_model = AcousticModelCRnn8Dropout(condition_size, classes_num, midfeat)
        else:
            self.velocity_model = None
            self.final_gru_ratio = 2

        self.reg_onset_gru = nn.GRU(
            input_size=classes_num * 2,
            hidden_size=256,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=True,
        )
        self.reg_onset_fc = nn.Linear(512, classes_num, bias=True)

        self.frame_gru = nn.GRU(
            input_size=classes_num * self.final_gru_ratio,
            hidden_size=256,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=True,
        )
        self.frame_fc = nn.Linear(512, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_gru(self.reg_onset_gru)
        init_gru(self.frame_gru)
        init_layer(self.reg_onset_fc)
        init_layer(self.frame_fc)

    def forward(self, waveform, condition):
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
        x = self.spec_layer(waveform)
        x = x.transpose(1,2) # (B, T, n_mels)
        x = torch.log(x+epsilon) 
        x = x.unsqueeze(1) # (B, 1, T, n_mels)
        
        spec = x

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        frame_output = self.frame_model(x, condition)  # (batch_size, time_steps, classes_num)
        reg_onset_output = self.reg_onset_model(x, condition)  # (batch_size, time_steps, classes_num)

        if self.reg_offset_model:
            reg_offset_output = self.reg_offset_model(x, condition)  # (batch_size, time_steps, classes_num)

        if self.velocity_model:
            velocity_output = self.velocity_model(x, condition)  # (batch_size, time_steps, classes_num)
            # Use velocities to condition onset regression.
            x = torch.cat((reg_onset_output, (reg_onset_output ** 0.5) * velocity_output.detach()), dim=2)
            (x, _) = self.reg_onset_gru(x)
            x = F.dropout(x, p=0.5, training=self.training, inplace=False)
            reg_onset_output = torch.sigmoid(self.reg_onset_fc(x))
        else:
            velocity_output = None

        # Use onsets and offsets to condition frame-wise classification
        if self.reg_offset_model:
            x = torch.cat((frame_output, reg_onset_output.detach(), reg_offset_output.detach()), dim=2)
        else:
            x = torch.cat((frame_output, reg_onset_output.detach()), dim=2)

        (x, _) = self.frame_gru(x)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)

        frame_output = torch.sigmoid(self.frame_fc(x))  # (batch_size, time_steps, classes_num)
        # (batch_size, time_steps, classes_num)

        output_dict = {
            'spec': spec,
            'reg_onset_output': reg_onset_output,
            'frame_output': frame_output,
        }
        if self.reg_offset_model:
            output_dict['reg_offset_output'] = reg_offset_output
        if self.velocity_model:
            output_dict['velocity_output'] = velocity_output

        return output_dict


class CNN(nn.Module):
    def __init__(self, frames_per_second, classes_num, modeling_velocity=None):
        super(CNN, self).__init__()

        sample_rate = SAMPLE_RATE
        window_size = 2048
        hop_size = sample_rate // frames_per_second
        mel_bins = 229
        fmin = 30
        fmax = sample_rate // 2

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        midfeat = 1792

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

        self.frame_model = AcousticModelCnn8Dropout(classes_num, midfeat)
        self.reg_onset_model = AcousticModelCnn8Dropout(classes_num, midfeat)
        self.reg_offset_model = AcousticModelCnn8Dropout(classes_num, midfeat)

        if modeling_velocity:
            self.velocity_model = AcousticModelCnn8Dropout(classes_num, midfeat)
        else:
            self.velocity_model = None

        self.reg_onset_fc = nn.Linear(classes_num * 2, classes_num, bias=True)

        self.frame_fc = nn.Linear(classes_num * 3, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.reg_onset_fc)
        init_layer(self.frame_fc)

    def forward(self, input):
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
        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        frame_output = self.frame_model(x)  # (batch_size, time_steps, classes_num)
        reg_onset_output = self.reg_onset_model(x)  # (batch_size, time_steps, classes_num)
        reg_offset_output = self.reg_offset_model(x)  # (batch_size, time_steps, classes_num)

        if self.velocity_model:
            velocity_output = self.velocity_model(x)  # (batch_size, time_steps, classes_num)
            # Use velocities to condition onset regression.
            x = torch.cat((reg_onset_output, (reg_onset_output ** 0.5) * velocity_output.detach()), dim=2)
            reg_onset_output = torch.sigmoid(self.reg_onset_fc(x))
        else:
            velocity_output = None

        # Use onsets and offsets to condition frame-wise classification
        x = torch.cat((frame_output, reg_onset_output.detach(), reg_offset_output.detach()), dim=2)
        frame_output = torch.sigmoid(self.frame_fc(x))  # (batch_size, time_steps, classes_num)
        # (batch_size, time_steps, classes_num)

        output_dict = {
            'reg_onset_output': reg_onset_output,
            'reg_offset_output': reg_offset_output,
            'frame_output': frame_output,
        }
        if self.velocity_model:
            output_dict['velocity_output'] = velocity_output

        return output_dict