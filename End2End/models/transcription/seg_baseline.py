import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn.init as init
import numpy as np
from torchaudio import transforms
from End2End.constants import BN_MOMENTUM, SAMPLE_RATE

epsilon=1e-10

def calculate_padding(input_size, kernel_size, stride):
    def calculate_padding_1D(input_size, kernel_size, stride):
        if (input_size%stride==0):
            pad = max(kernel_size-stride, 0)
        else:
            pad = max(kernel_size-(input_size%stride), 0)

        return pad
    
    if type(kernel_size) != tuple:
        kernel_size_1 = kernel_size
        kernel_size_2 = kernel_size
    else:
        kernel_size_1 = kernel_size[0]
        kernel_size_2 = kernel_size[1]      

    if type(stride) != tuple:
        stride_1 = stride
        stride_2 = stride
    else:
        stride_1 = stride[0]
        stride_2 = stride[1]

    padding1 = calculate_padding_1D(input_size[0], kernel_size_1, stride_1)
    padding2 = calculate_padding_1D(input_size[1], kernel_size_2, stride_2)

    pad_top = padding1//2
    pad_bottom = padding1 - pad_top
    pad_left = padding2//2
    pad_right = padding2 - pad_left
    
    return (pad_left,pad_right,pad_top,pad_bottom)  

def transpose_padding_same(x, input_shape, stride):
    """
    Trying to implement padding='SAME' as in tensorflow for the Conv2dTranspose layer.
    It is basically trying to remove paddings from the output
    """
    
    input_shape = torch.tensor(input_shape[2:])*torch.tensor(stride)
    output_shape = torch.tensor(x.shape[2:])
    
    if torch.equal(input_shape,output_shape):
        print(f'same, no need to do anything')
        pass
    else:
        padding_remove = (output_shape-input_shape)
        left = torch.div(padding_remove, 2, rounding_mode='floor')
        right = torch.div(padding_remove, 2, rounding_mode='floor') + padding_remove%2    
#         left = padding_remove//2
#         right = padding_remove//2+padding_remove%2
        
    return x[:,:,left[0]:-right[0],left[1]:-right[1]]

def SAME_padding(x, ksize, stride):
    padding = calculate_padding(x.shape[2:], ksize, stride)
    return F.pad(x, padding) 


class Conv_Block(nn.Module):
    def __init__(self, inp, out, ksize, stride=(2,2), dilation_rate=1, dropout_rate=0.4):
        super().__init__()
        self.ksize = ksize
        self.stride = stride
        self.stride_conv2 = 1
        self.ksize_skip = 1
        
        padding=0 # We don't pad with the Conv2d class, we use F.pad to pad instead
        
        self.conv1 = nn.Conv2d(inp,out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation_rate)
        self.bn1 = nn.BatchNorm2d(inp)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.conv2 = nn.Conv2d(out, out, kernel_size=ksize, stride=self.stride_conv2, padding=padding, dilation=dilation_rate)
        self.bn2 = nn.BatchNorm2d(out)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        if self.stride!=(1,1):
            self.conv_skip = nn.Conv2d(inp, out, kernel_size=self.ksize_skip, stride=stride, padding=padding)
        

    def forward(self, x):
        skip = x # save a copy for the skip connection later
        
        x = self.bn1(torch.relu(x))
        x = self.dropout1(x)
        
        # Calculating padding corresponding to 'SAME' in tf
        x = SAME_padding(x, self.ksize, self.stride)
        x = self.conv1(x)
        
        x = self.bn2(torch.relu(x))
        x = self.dropout2(x)
        
        # Calculating padding corresponding to 'SAME' in tf
        x = SAME_padding(x, self.ksize, self.stride_conv2)
        x = self.conv2(x)
        
        if self.stride!=(1,1):
        # Calculating padding corresponding to 'SAME' in tf
            skip = SAME_padding(skip, self.ksize_skip, self.stride)
            # Padding is mostly 0 so far, comment it out first
            skip = self.conv_skip(skip)
        x = x + skip # skip connection
        
        return x


class transpose_conv_block(nn.Module):
    def __init__(self, inp, out, ksize, stride=(2,2), dropout_rate=0.4):
        super().__init__()
        
        self.stride = stride
        self.ksize = ksize
        padding=0 # We don't pad with the Conv2d class, we use F.pad to pad instead
        
        self.conv1 = nn.Conv2d(inp,out, kernel_size=ksize, stride=(1,1), padding=padding)
        self.bn1 = nn.BatchNorm2d(inp)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        
        self.conv2 = nn.ConvTranspose2d(out, out, kernel_size=ksize, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(out)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.conv_skip = nn.ConvTranspose2d(inp, out, kernel_size=1, stride=stride, padding=padding)
        

    def forward(self, x, shape):
        skip = x # save a copy for the skip connection later
        input_shape_skip = skip.shape # will be used as in the transpose padding later
        
        x = self.bn1(torch.relu(x))
        x = self.dropout1(x)
        x = SAME_padding(x, self.ksize, (1,1))
        x = self.conv1(x)
        
#         transpose_conv1 = torch.Size([1, 128, 40, 15])        
        
        x = self.bn2(torch.relu(x))
        x = self.dropout2(x)
        input_shape = x.shape
        x = self.conv2(x)
        x = transpose_padding_same(x, input_shape, self.stride)
        
        # Removing extra pixels induced due to ConvTranspose
        if x.shape[2]>shape[2]:
            x = x[:,:,:-1,:]
        if x.shape[3]>shape[3]:
            x = x[:,:,:,:-1]           
        
#         transpose_conv2 = torch.Size([1, 128, 83, 35])        
        
        if self.stride!=(1,1):
            # Check keras about the transConv output shape
            skip = self.conv_skip(skip, output_size=x.shape) # make output size same as x
#             skip = transpose_padding_same(skip, input_shape_skip, self.stride)
    
        x = x + skip # skip connection
        
        return x
    
class Decoder_Block(nn.Module):
    def __init__(self,
                 input_channels,
                 encoder_channels,
                 hidden_channels,
                 output_channels,
                 dropout_rate=0.4):
        super().__init__()
        
        # Again, not using Conv2d to calculate the padding,
        # use F.pad to obtain a more general padding under forward
        self.ksize = (1,1)
        self.stride = (1,1)
        self.layer1a = nn.Conv2d(input_channels+encoder_channels, hidden_channels, kernel_size=self.ksize, stride=self.stride) # the channel dim for feature
        self.bn = nn.BatchNorm2d(input_channels)
        self.bn_en = nn.BatchNorm2d(encoder_channels)
        self.dropout1 = nn.Dropout(dropout_rate)    
        self.layer1b = transpose_conv_block(input_channels, output_channels, (3,3), (2,2))
        

    def forward(self, x, encoder_output, encoder_shape):
        skip = x # save a copy for the skip connection later
        
        x = self.bn(torch.relu(x))

        en_l = self.bn_en(torch.relu(encoder_output))
        
        x = torch.cat((x, en_l), 1)
        x = self.dropout1(x)
        
        x = SAME_padding(x, self.ksize, self.stride)
        x = self.layer1a(x)
        x = x + skip
        
        x = self.layer1b(x, encoder_shape)
        
        return x
    
class MutliHeadAttention2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), groups=1, bias=False):
        """kernel_size is the 2D local attention window size"""

        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        # Padding should always be (kernel_size-1)/2
        # Isn't it?
        self.padding_time = (kernel_size[0]-1)//2
        self.padding_freq = (kernel_size[1]-1)//2
        self.groups = groups

        # Make sure the feature dim is divisible by the n_heads
        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        # Relative position encoding
        self.rel_t = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size[0], 1), requires_grad=True)
        self.rel_f = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size[1]), requires_grad=True)

        # Increasing the channel deapth (feature dim) with Conv2D
        # kernel_size=1 such that it expands only the feature dim
        # without affecting other dimensions
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding_freq, self.padding_freq, self.padding_time, self.padding_time])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)
        
        k_out = k_out.unfold(2, self.kernel_size[0], self.stride[0]).unfold(3, self.kernel_size[1], self.stride[1])
        # (batch, channels, H, W, H_local_w, W_local_w) 
        
        v_out = v_out.unfold(2, self.kernel_size[0], self.stride[0]).unfold(3, self.kernel_size[1], self.stride[1])
        # (batch, channels, H, W, H_local_w, W_local_w) 

        k_out_t, k_out_f = k_out.split(self.out_channels // 2, dim=1)
        
        k_out = torch.cat((k_out_t + self.rel_t, k_out_f + self.rel_f), dim=1) # relative position?

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        # (batch, n_heads, feature_per_head, H, W, local H X W)
        
        # expand the last dimension s.t. it can multiple with the local att window
        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)
        # (batch, n_heads, feature_per_head, H, W, 1)

        # Alternative way to express dot product
        # same as k_out = k_out.permute(0,1,3,4,2,5)
        # and then energy = torch.matmul(q_out,k_out) 
        energy = (q_out * k_out).sum(dim=2, keepdim=True)
        
        attention = F.softmax(energy, dim=-1)
        # (batch, n_heads, 1, H, W, local HXW)
        
        out = attention*v_out
        # (batch, n_heads, feature_per_head, H, W, local HXW)
        # (batch, c, H, W)
        
        return out.sum(-1).flatten(1,2), attention.squeeze(2)

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_t, 0, 1)
        init.normal_(self.rel_f, 0, 1)            

class Encoder(nn.Module):
    def __init__(self,
                 input_size,
                 feature_num=128,
                 timesteps=256,
                 multi_grid_layer_n=1,
                 multi_grid_n=3,
                 ch_num=1,
                 prog=False,
                 dropout_rate=0.4,
                 out_class=2):
        super().__init__()
        
        # Parameters for the encoding layer
        en_kernel_size = (7,7)
        en_stride = (1,1)
        # Again, not using Conv2d to calculate the padding,
        # use F.pad to obtain a more general padding under forward
        self.en_padding = calculate_padding(input_size, en_kernel_size, en_stride)
        # Instead of using Z, it should be using Z_f and Z_q
        # But for the sake of this experiment, 
        self.encoding_layer = nn.Conv2d(1, 2**5, kernel_size=en_kernel_size, stride=en_stride, padding=0)

        self.layer1a = Conv_Block(2**5, 2**5, ksize=(3,3), stride=(2,2), dropout_rate=dropout_rate)
        self.layer1b = Conv_Block(2**5, 2**5, ksize=(3,3), stride=(1,1), dropout_rate=dropout_rate)
        
        self.layer2a = Conv_Block(2**5, 2**6, ksize=(3,3), stride=(2,2), dropout_rate=dropout_rate)
        self.layer2b = Conv_Block(2**6, 2**6, ksize=(3,3), stride=(1,1), dropout_rate=dropout_rate)
        self.layer2c = Conv_Block(2**6, 2**6, ksize=(3,3), stride=(1,1), dropout_rate=dropout_rate)
        
        self.layer3a = Conv_Block(2**6, 2**7, ksize=(3,3), stride=(2,2), dropout_rate=dropout_rate)
        self.layer3b = Conv_Block(2**7, 2**7, ksize=(3,3), stride=(1,1), dropout_rate=dropout_rate)
        self.layer3c = Conv_Block(2**7, 2**7, ksize=(3,3), stride=(1,1), dropout_rate=dropout_rate)
        self.layer3d = Conv_Block(2**7, 2**7, ksize=(3,3), stride=(1,1), dropout_rate=dropout_rate)
        
        self.layer4a = Conv_Block(2**7, 2**8, ksize=(3,3), stride=(2,2), dropout_rate=dropout_rate)
        self.layer4b = Conv_Block(2**8, 2**8, ksize=(3,3), stride=(1,1), dropout_rate=dropout_rate)
        self.layer4c = Conv_Block(2**8, 2**8, ksize=(3,3), stride=(1,1), dropout_rate=dropout_rate)
        self.layer4d = Conv_Block(2**8, 2**8, ksize=(3,3), stride=(1,1), dropout_rate=dropout_rate)
        self.layer4e = Conv_Block(2**8, 2**8, ksize=(3,3), stride=(1,1), dropout_rate=dropout_rate)

        

    def forward(self, x):
        skip = x # save a copy for the skip connection later
        original_shape = x.shape
        
        x = F.pad(x, self.en_padding)
        x = self.encoding_layer(x)
        x = self.layer1a(x)
        x = self.layer1b(x)
        en_l1 = x
        shape1 = x.shape
        x = self.layer2a(x)
        x = self.layer2b(x)
        x = self.layer2c(x)
        shape2 = x.shape
        en_l2 = x

        x = self.layer3a(x)
        x = self.layer3b(x)
        x = self.layer3c(x)
        x = self.layer3d(x)
        shape3 = x.shape
        en_l3 = x

        x = self.layer4a(x)
        x = self.layer4b(x)
        x = self.layer4c(x)
        x = self.layer4d(x)
        x = self.layer4e(x)
        shape4 = x.shape
        en_l4 = x
        
        # en_l4 and shape4 could not be used inside the decoder, that's why they are omitted
        return x, (en_l1, en_l2, en_l3), (original_shape, shape1, shape2, shape3)
    
    
class Decoder(nn.Module):
    def __init__(self,
                 dropout_rate=0.4):
        super().__init__()
        
        self.de_layer1 = Decoder_Block(2**7, 2**7, 2**7, 2**6, dropout_rate)
        self.de_layer2 = Decoder_Block(2**6, 2**6, 2**6, 2**6, dropout_rate)
        self.de_layer3 = Decoder_Block(2**6, 2**5, 2**6, 2**6, dropout_rate)
        

    def forward(self, x, encoder_outputs, encoder_shapes):
        x = self.de_layer1(x, encoder_outputs[-1], encoder_shapes[-2])
        x = self.de_layer2(x, encoder_outputs[-2], encoder_shapes[-3])
        x = self.de_layer3(x, encoder_outputs[-3], encoder_shapes[-4]) # Check this
        return x
    
    
class Semantic_Segmentation(nn.Module):
    def __init__(self, cfg, out_class=2, dropout_rate=0.4):
        super().__init__()
        
        self.spec_layer = transforms.MelSpectrogram(**cfg.transcription.feature.STFT)
#         self.bn0 = nn.BatchNorm2d(cfg.feature.STFT.n_mels, momentum=BN_MOMENTUM) # for normalizing the spectrograms        
        shpae = (1001,229)
        self.encoder = Encoder(shpae, dropout_rate=dropout_rate) # hard coding the shape here, since our spectrogram has a fixed size
        self.attention_layer1 = MutliHeadAttention2D(256, 64, kernel_size=(17,17), stride=(1,1), groups=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.attention_layer2 = MutliHeadAttention2D(64, 128, kernel_size=(17,17), stride=(1,1), groups=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        
        # L218-221 of the original code
        # Few layers before the Decoder part        
        self.layer0a = nn.Conv2d(384, 2**8, (1,1), (1,1))
        self.layer0b = transpose_conv_block(2**8, 2**7, (3,3), (2,2))
        
        self.decoder = Decoder(dropout_rate=dropout_rate)
        
        # Last few layers that determines the output
        self.bn_last = nn.BatchNorm2d(2**6)
        self.dropout_last = nn.Dropout(dropout_rate)
        self.conv_last = nn.Conv2d(2**6, out_class, (1,1), (1,1))
        
        self.inference_model = nn.Linear(229, 88)
        self.frames_per_second = cfg.frames_per_second
        self.classes_num = 88 # hard code it for correct evaluation at the moment
        
    def forward(self, waveform):
        
        x = self.spec_layer(waveform)
        x = x.transpose(1,2) # (B, T, n_mels)
        x = torch.log(x+epsilon) 
        
        x = x.unsqueeze(1) # (B, 1, T, n_mels)
        
        spec = x

#         x = x.transpose(1, 3)
#         x = self.bn0(x)
#         x = x.transpose(1, 3)        
        
        x, encoder_outputs, encoder_shapes = self.encoder(x)
        en_l4 = x # Will be appened with the attention output and decoder later

        # Two layers of self-attention 
        x,_ = self.attention_layer1(en_l4)
        x = self.bn1(torch.relu(x))

        x, _ = self.attention_layer2(x)
        x = self.bn2(torch.relu(x))
        x = torch.cat((en_l4, x),1) # L216

        # L218-221 of the original code
        # Few layers before the Decoder part
        x = SAME_padding(x, (1,1), (1,1))
        x = self.layer0a(x)
        x = x + en_l4   
        x = self.layer0b(x, encoder_shapes[-1]) # Transposing back to the Encoder shape
        
        # Decoder part
        x = self.decoder(x, encoder_outputs, encoder_shapes)   
        
        # Last few layers for the output block
        x = self.bn_last(torch.relu(x))
        x = self.dropout_last(x)
        x = self.conv_last(x)
        
        # We use a Linear layer as the inference model here
        x = x.squeeze(1) # remove the channel dim
        x = self.inference_model(x)
        x = torch.sigmoid(x)
        
        output_dict = {
            'spec': spec,
            'reg_onset_output': x,
            'frame_output': x,
        }        
        
        
        return output_dict