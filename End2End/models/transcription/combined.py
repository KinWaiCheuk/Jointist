import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import transforms
from torchaudio.functional import amplitude_to_DB

import End2End.models.transcription.acoustic as Acoustic
from End2End.models.utils import init_bn, init_gru, init_layer
from End2End.constants import BN_MOMENTUM, SAMPLE_RATE

epsilon=1e-10

class BackEnd(nn.Module):
    def __init__(self, backend_cfg, classes_num):
        super().__init__()

        self.acoustic = getattr(Acoustic, backend_cfg.acoustic.type)(**backend_cfg.acoustic.args)
        self.acoustic_dropout = nn.Dropout(backend_cfg.acoustic_dropout)
        self.language = getattr(nn, backend_cfg.language.type)(**backend_cfg.language.args)
        self.language_dropout = nn.Dropout(backend_cfg.language_dropout)        

        self.fc5 = nn.Linear(self.acoustic.freq_dim*self.acoustic.out_channels, backend_cfg.acoustic_dim, bias=False)
        self.bn5 = nn.BatchNorm1d(backend_cfg.acoustic_dim, momentum=BN_MOMENTUM)

        self.fc = nn.Linear(backend_cfg.language_dim, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_layer(self.fc5)
        init_bn(self.bn5)
        init_gru(self.language)
        init_layer(self.fc)

    def forward(self, input, condition):
        r"""
        Args:
            input: (batch_size, channels_num, time_steps, freq_bins)

        Outputs:
            output: (batch_size, time_steps, classes_num)
        """
        x = self.acoustic(input, condition)  # (batch, ch, time, freq)

        x = x.transpose(1, 2).flatten(start_dim=2)  # -> (b, t, ch, f) -> (b, t, ch * f)
        x = F.relu(self.bn5(self.fc5(x).transpose(1, 2)).transpose(1, 2))
        x = self.acoustic_dropout(x)  # (b, t, ch * f)

        (x, _) = self.language(x)
        x = self.language_dropout(x)
        output = torch.sigmoid(self.fc(x))
        return output

class Original(nn.Module):
    def __init__(self, cfg, frames_per_second, classes_num, modeling_offset, modeling_velocity=None):
        super().__init__()

        self.frames_per_second = frames_per_second
        self.classes_num = classes_num
        
        self.spec_layer = transforms.MelSpectrogram(**cfg.transcription.feature.STFT)

        self.bn0 = nn.BatchNorm2d(cfg.transcription.feature.STFT.n_mels, momentum=BN_MOMENTUM)

        self.frame_model = BackEnd(cfg.transcription.backend, classes_num)
        self.reg_onset_model = BackEnd(cfg.transcription.backend, classes_num)
        
     

        if modeling_offset:
            self.reg_offset_model = BackEnd(cfg.transcription.backend, classes_num)
            self.final_gru_ratio = 3
        else:
            self.reg_offset_model = None
            self.final_gru_ratio = 2

        if modeling_velocity:
            self.velocity_model = BackEnd(cfg.transcription.backend, classes_num)
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
        else:
            self.velocity_model = None
            self.final_gru_ratio = 2
            
            
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

#         self.init_weight()

#     def init_weight(self):
#         init_bn(self.bn0)
# #         init_gru(self.reg_onset_gru)
#         init_gru(self.frame_gru)
# #         init_layer(self.reg_onset_fc)
#         init_layer(self.frame_fc)

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
    
    
class FrameOnly(nn.Module):
    def __init__(self, cfg, frames_per_second, classes_num, modeling_offset, modeling_velocity=None):
        super().__init__()

        self.frames_per_second = frames_per_second
        self.classes_num = classes_num
        
        self.spec_layer = transforms.MelSpectrogram(**cfg.feature.STFT)

        self.bn0 = nn.BatchNorm2d(cfg.feature.STFT.n_mels, momentum=BN_MOMENTUM)

        self.frame_model = BackEnd(cfg.transcription.backend, classes_num)
        
        self.velocity_model = None
        self.final_gru_ratio = 1
            
            
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

#         self.init_weight()

#     def init_weight(self):
#         init_bn(self.bn0)
# #         init_gru(self.reg_onset_gru)
#         init_gru(self.frame_gru)
# #         init_layer(self.reg_onset_fc)
#         init_layer(self.frame_fc)

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

        velocity_output = None

        (x, _) = self.frame_gru(frame_output)
        x = F.dropout(x, p=0.5, training=self.training, inplace=False)

        frame_output = torch.sigmoid(self.frame_fc(x))  # (batch_size, time_steps, classes_num)
        # (batch_size, time_steps, classes_num)

        output_dict = {
            'spec': spec,
            'reg_onset_output': frame_output,
            'frame_output': frame_output,
        }

        return output_dict