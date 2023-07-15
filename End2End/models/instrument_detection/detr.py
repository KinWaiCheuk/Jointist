import torch
import torch.nn as nn
from torchaudio import transforms
from torchaudio.functional import amplitude_to_DB
from torchvision.models import resnet50
import torch.nn.functional as F
import math
from End2End.models.utils import init_layer, init_bn, Normalization
from End2End.models.transformer import DETR_Transformer
from End2End.models.position_encoding import PositionEmbeddingSine

import sys

epsilon=1e-10 # small number for taking log on spectrograms


def pos_encoder(max_seq_len, d_model):
    pe = torch.zeros(max_seq_len, d_model)
    for pos in range(max_seq_len):
        for i in range(0, d_model, 2):
            pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i) / d_model)))
            pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
    return pe

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
#         self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
#         self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
#         self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
#         self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
#         self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
#         self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

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
    
    
    
class Cnn14_DETR_transformer(nn.Module):
    """
    Combining Jointist CNN together with Transformer
    """
    def __init__(self, num_classes, spec_args, num_Q, learnable_pos, feature_weight, max_pos=50, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):

        super().__init__()
        self.num_classes = num_classes
        self.feature_weight = feature_weight
        self.spec_layer = transforms.MelSpectrogram(**spec_args.STFT)
#         self.dB_args = spec_args.dB_args
#         self.normalizer = Normalization('imagewise')
        
        self.bn0 = nn.BatchNorm2d(spec_args.STFT.n_mels)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        
        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)        
        
        
        # create a default PyTorch transformer
        self.transformer = Transformer(
            d_model=hidden_dim,
            dropout=0.1,
            nhead=nheads,
            dim_feedforward=2048,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            normalize_before=False,
            return_intermediate_dec=True
            )
        
        self.pos_layer = PositionEmbeddingSine()

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes)

        # output positional encodings (object queries)
#         self.query_pos = nn.Parameter(torch.rand(num_Q, hidden_dim))
        self.query_embed = nn.Embedding(num_Q, hidden_dim)        

        # spatial positional encodings
#         if learnable_pos:
#             # note that in baseline DETR we use sine positional encodings
#             self.register_parameter('row_embed', nn.Parameter(torch.rand(max_pos, hidden_dim // 2)))
#             self.register_parameter('col_embed', nn.Parameter(torch.rand(max_pos, hidden_dim // 2)))
#         else:
#             self.register_buffer('row_embed', pos_encoder(max_pos, hidden_dim // 2))
#             self.register_buffer('col_embed', pos_encoder(max_pos, hidden_dim // 2))        
        
#         self.fc1 = nn.Linear(2048, 2048, bias=True)
#         self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
#         self.fc2 = nn.Linear(2048, 2048, bias=True)
#         self.fc_audiocount = nn.Linear(2048, 1, bias=True)        

#         self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
#         init_layer(self.fc1)
#         init_layer(self.fc_audioset)

    def forward(self, waveform):
        """
        Input: (batch_size, data_length)"""
        x = self.spec_layer(waveform)
        x = x.transpose(1,2) # (B, T, n_mels)
        x = torch.log(x+epsilon) 
        x = x.unsqueeze(1) # (B, 1, T, n_mels)
        
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
#         x = torch.mean(x, dim=3) # (8, 2048, 31)
        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x) # (B, hidde) (8, 256, 32, 8) 
#         # construct positional encodings
        B, _, H, W = h.shape
        pos = self.pos_layer(h)
        hs = self.transformer(h, None, self.query_embed.weight, pos)[0]
        # (num_decoder_layers, B, Q, hidden_dim)      
        
        h = hs[-1] # (B, Q, hidden_dim)
            
        aux_outputs = []
        for i in hs[:-1]:
            aux_outputs.append({'pred_logits': self.linear_class(i)})
            
    
        return {'pred_logits': self.linear_class(h),
                'aux_outputs': aux_outputs,
                'spec': spec}
    
    
class Cnn14Transformer(nn.Module):
    """
    Combining Jointist CNN together with Transformer
    """
    def __init__(self, num_classes, spec_args, num_Q, learnable_pos, feature_weight, max_pos=50, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):

        super().__init__()
        self.num_classes = num_classes
        self.feature_weight = feature_weight
        self.spec_layer = transforms.MelSpectrogram(**spec_args.STFT)
#         self.dB_args = spec_args.dB_args
#         self.normalizer = Normalization('imagewise')
        
        self.bn0 = nn.BatchNorm2d(spec_args.STFT.n_mels)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        
        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)        
        
        
        # create a default PyTorch transformer
        if num_encoder_layers!=0:
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nheads)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)        
#         self.transformer = nn.Transformer(
#             hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nheads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(num_Q, hidden_dim))

        # spatial positional encodings
        if learnable_pos:
            # note that in baseline DETR we use sine positional encodings
            self.register_parameter('row_embed', nn.Parameter(torch.rand(max_pos, hidden_dim // 2)))
            self.register_parameter('col_embed', nn.Parameter(torch.rand(max_pos, hidden_dim // 2)))
        else:
            self.register_buffer('row_embed', pos_encoder(max_pos, hidden_dim // 2))
            self.register_buffer('col_embed', pos_encoder(max_pos, hidden_dim // 2))        
        
#         self.fc1 = nn.Linear(2048, 2048, bias=True)
#         self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
#         self.fc2 = nn.Linear(2048, 2048, bias=True)
#         self.fc_audiocount = nn.Linear(2048, 1, bias=True)        

#         self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
#         init_layer(self.fc1)
#         init_layer(self.fc_audioset)

    def forward(self, waveform):
        """
        Input: (batch_size, data_length)"""
        x = self.spec_layer(waveform)
        x = x.transpose(1,2) # (B, T, n_mels)
        x = torch.log(x+epsilon) 
        x = x.unsqueeze(1) # (B, 1, T, n_mels)
        
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
#         x = torch.mean(x, dim=3) # (8, 2048, 31)
        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x) # (B, hidde) (8, 256, 32, 8) 
#         # construct positional encodings
        B, _, H, W = h.shape
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1).repeat(1,B,1) # (H*W, B, features)=(256, 8, 256)
                 
        if hasattr(self, 'transformer_encoder'):
            src = pos + self.feature_weight * h.flatten(2).permute(2, 0, 1) # (H*W, B, features)=(256, 8, 256)
            memory = self.transformer_encoder(src, mask=None, src_key_padding_mask=None)
            memory = pos + self.feature_weight * memory
        else:
            memory = pos + self.feature_weight * h.flatten(2).permute(2, 0, 1)
#         memory = self.transformer.encoder(src, mask=None, src_key_padding_mask=None) # (num_Q, B, features)=(256, 8, 256)  

        tgt = self.query_pos.unsqueeze(1).repeat(1,B,1) # (num_Q, B, features)=(Q, 8, 256)
        hs = []
        for layer in self.transformer_decoder.layers:
            tgt = layer(tgt,
                        memory,
                        tgt_mask=None,
                        memory_mask=None,
                        tgt_key_padding_mask=None,
                        memory_key_padding_mask=None)
            hs.append(tgt)
        
        h = hs[-1].transpose(0,1) # (B, Q, features)
            
        # propagate through the transformer
#         h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
#                              self.query_pos.unsqueeze(1).repeat(1,B,1)).transpose(0, 1)
        # finally project transformer outputs to class labels and bounding boxes
        aux_outputs = []
        for i in hs[:-1]:
            aux_outputs.append({'pred_logits': self.linear_class(i.transpose(0,1))})
            
    
        return {'pred_logits': self.linear_class(h),
                'aux_outputs': aux_outputs,
                'spec': spec}

    
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