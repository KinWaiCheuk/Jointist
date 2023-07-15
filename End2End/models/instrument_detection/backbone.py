import torch
import torch.nn as nn
from torchaudio import transforms
from torchaudio.functional import amplitude_to_DB
from torchvision.models import resnet50
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
import torch.nn.functional as F
import math
from End2End.models.utils import init_layer, init_bn, Normalization
from End2End.util.misc import is_main_process

import sys

class AcousticModelCnn8Dropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout
        # condition_size = 167

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=48)
        self.conv_block2 = ConvBlock(in_channels=48, out_channels=64)
        self.conv_block3 = ConvBlock(in_channels=64, out_channels=96)
        self.conv_block4 = ConvBlock(in_channels=96, out_channels=128)
        
        self.num_channels = 128
        self.F = 14

    def forward(self, input):
        r"""
        Args:
            input: (batch_size, channels_num, time_steps, freq_bins)

        Outputs:
            output: (batch_size, time_steps, classes_num)
        """
        x = self.conv_block1(input, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=self.dropout)
        x = self.conv_block2(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=self.dropout)
        x = self.conv_block3(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=self.dropout)
        x = self.conv_block4(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=self.dropout)  # (batch, ch, time, freq)

        return x

    
class CNN8(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout
        # condition_size = 167

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=48)
        self.conv_block2 = ConvBlock(in_channels=48, out_channels=64)
        self.conv_block3 = ConvBlock(in_channels=64, out_channels=96)
        self.conv_block4 = ConvBlock(in_channels=96, out_channels=128)
        
        self.num_channels = 128
        self.F = 14

    def forward(self, input):
        r"""
        Args:
            input: (batch_size, channels_num, time_steps, freq_bins)

        Outputs:
            output: (batch_size, time_steps, classes_num)
        """
        x = self.conv_block1(input, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=self.dropout)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=self.dropout)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=self.dropout)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=self.dropout)  # (batch, ch, time, freq)

        return x    
    
class CNN14(nn.Module):
    """
    Combining Jointist CNN together with Transformer
    """
    def __init__(self, n_mels, channel_list, dropout):

        super().__init__()       
        self.bn0 = nn.BatchNorm2d(n_mels)

        cnn_layers = []
        previous_channel = 1
        for channel in channel_list:
            cnn_layers.append(ConvBlock(in_channels=previous_channel, out_channels=channel))
            cnn_layers.append(nn.Dropout(dropout))
            previous_channel=channel
        self.feature_extractor = nn.Sequential(*cnn_layers)
        self.num_channels = channel
        self.F = 3

    def init_weight(self):
        init_bn(self.bn0)

    def forward(self, x):
        """
        Input: Spectragrams (B, 1, T, n_mels)"""
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.feature_extractor(x) # (8, 2048, 31, 7)          
    
        return x
    
    
class CNN14_less_pooling(nn.Module):
    """
    Combining Jointist CNN together with Transformer
    """
    def __init__(self, n_mels, channel_list, dropout, num_pooling, pool_first):

        super().__init__()       
        self.bn0 = nn.BatchNorm2d(n_mels)
        self.num_pooling = num_pooling
        self.pool_first = pool_first
        
        cnn_layers = []
        previous_channel = 1
        self.num_layers = len(channel_list)
        for channel in channel_list:
            cnn_layers.append(ConvBlock(in_channels=previous_channel, out_channels=channel))                 
            cnn_layers.append(nn.Dropout(dropout))
            previous_channel=channel
        self.feature_extractor = nn.Sequential(*cnn_layers)
        self.num_channels = channel
        self.F = n_mels//2**num_pooling

    def init_weight(self):
        init_bn(self.bn0)

    def forward(self, x):
        """
        Input: Spectragrams (B, 1, T, n_mels)"""
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        conv_counter = 0
        for layer in self.feature_extractor.children():
            if isinstance(layer, ConvBlock):
                if self.pool_first:
                    if conv_counter < self.num_pooling: 
                        x = layer(x, pool_type='avg')
                    else: # Don't do pooling
                        x = layer(x, pool_type='none')
                else:
                    if conv_counter > self.num_layers - self.num_pooling:
                        x = layer(x, pool_type='avg')
                    else: # Don't do pooling
                        x = layer(x, pool_type='none')
                conv_counter += 1
            else:
                x = layer(x)
                
#         x = self.feature_extractor(x) # (8, 2048, 31, 7)          
    
        return x    
    
    
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
        elif pool_type == 'none':
            pass # No pooling
        else:
            raise Exception('Incorrect argument!')

        return x
    
class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, x):
        xs = self.body(x) # output is dictionary 
        return xs['0'] # extract tensor from dictionary
    
class Resnet50(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        name = 'resnet50'
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=False, norm_layer=FrozenBatchNorm2d)
        backbone.conv1 = torch.nn.Conv1d(1, 64, (7, 7), (2, 2), (3, 3), bias=False) # replacing the 3 channels input to 1 channel input
        num_channels = 2048
        self.F = 15 # for auto infering CNN output shape later
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)
        
        
class Resnet50(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        name = 'resnet101'
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=False, norm_layer=FrozenBatchNorm2d)
        backbone.conv1 = torch.nn.Conv1d(1, 64, (7, 7), (2, 2), (3, 3), bias=False) # replacing the 3 channels input to 1 channel input
        num_channels = 2048
        self.F = 15 # for auto infering CNN output shape later
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)        
        
class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias        
        

        