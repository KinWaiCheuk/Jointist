import torch
import torch.nn as nn
from torchaudio import transforms
from torchaudio.functional import amplitude_to_DB
from torchvision.models import resnet50
import torch.nn.functional as F
import math
from End2End.models.utils import init_layer, init_bn, Normalization
from End2End.models.transformer import DETR_Transformer
from End2End.models.position_encoding import PositionEmbeddingSine, PositionEmbeddingSinev2
import numpy as np
from End2End.models.instrument_detection.backbone import ConvBlock

import sys

epsilon=1e-10 # small number for taking log on spectrograms

  
    
class CombinedModel_H(nn.Module):
    """
    Combining Jointist CNN together with Transformer
    """
    def __init__(self, model_args, backbone, transformer, spec_args):

        super().__init__()
        self.num_classes = model_args.args.num_classes
        self.feature_weight = model_args.args.feature_weight
        self.spec_layer = transforms.MelSpectrogram(**spec_args.STFT)
        # decalre a backbone
        self.backbone = backbone
        # declare a transformer        
        self.transformer=transformer        
        hidden_dim = self.transformer.d_model
        # create conversion layer to connect the backbone to transformer
        self.conv = nn.Conv2d(self.backbone.num_channels, hidden_dim, 1)        
        self.pos_layer = PositionEmbeddingSine(**model_args.positional)

        # prediction heads, one extra class for predicting empty slots
        self.linear_class = nn.Linear(hidden_dim, self.num_classes)
        self.query_embed = nn.Embedding(model_args.args.num_Q, hidden_dim)        


    def forward(self, waveform):
        """
        Input: (batch_size, data_length)"""
        x = self.spec_layer(waveform)
        x = x.transpose(1,2) # (B, T, n_mels)
        x = torch.log(x+epsilon) 
        x = x.unsqueeze(1) # (B, 1, T, n_mels)
        spec = x        
        
        x = self.backbone(x)
        backbone_feat = x
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
                'spec': spec,
                'backbone_feat': backbone_feat}
    
    
class CombinedModel_S(nn.Module):
    """
    Combining Jointist CNN together with Transformer
    """
    def __init__(self, model_args, backbone, transformer, spec_args):

        super().__init__()
        self.num_classes = model_args.args.num_classes
        self.feature_weight = model_args.args.feature_weight
        self.spec_layer = transforms.MelSpectrogram(**spec_args.STFT)
        # decalre a backbone
        self.backbone = backbone
        # declare a transformer        
        self.transformer=transformer        
        hidden_dim = self.transformer.d_model
        # create conversion layer to connect the backbone to transformer
        self.conv = nn.Conv2d(self.backbone.num_channels, hidden_dim, 1)        
        self.pos_layer = PositionEmbeddingSine(**model_args.positional)

        # prediction heads, one extra class for predicting non-empty slots
#         self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_class = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(model_args.args.num_Q, hidden_dim)        


    def forward(self, waveform):
        """
        Input: (batch_size, data_length)"""
        x = self.spec_layer(waveform)
        x = x.transpose(1,2) # (B, T, n_mels)
        x = torch.log(x+epsilon) 
        x = x.unsqueeze(1) # (B, 1, T, n_mels)
        spec = x        
        
        x = self.backbone(x)
        backbone_feat = x
        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x) # (B, hidde) (8, 256, 32, 8) / (32, 128, 1001, 14) / (32, 128, 62, 14)
#         # construct positional encodings
        B, _, H, W = h.shape
        pos = self.pos_layer(h)
        hs = self.transformer(h, None, self.query_embed.weight, pos)[0]
        # (num_decoder_layers, B, Q, hidden_dim)      
        
        h = hs[-1] # (B, Q, hidden_dim)
        
        h = F.dropout(h, p=0.5)
#         h = F.relu_(self.fc1(h))
            
        aux_outputs = []
        for i in hs[:-1]:
            i = F.dropout(i, p=0.5)
#             i = F.relu_(self.fc1(i))
            aux_outputs.append({'pred_logits': self.linear_class(i).flatten(1)})
            
    
        return {'pred_logits': self.linear_class(h).flatten(1),
                'aux_outputs': aux_outputs,
                'spec': spec,
                'backbone_feat': backbone_feat}
    
    
class CombinedModel_Sv2(nn.Module):
    """
    Combining Jointist CNN together with Transformer
    """
    def __init__(self, model_args, backbone, transformer, spec_args):

        super().__init__()
        self.num_classes = model_args.args.num_classes
        self.feature_weight = model_args.args.feature_weight
        self.spec_layer = transforms.MelSpectrogram(**spec_args.STFT)
        # decalre a backbone
        self.backbone = backbone
        # declare a transformer        
        self.transformer=transformer        
        hidden_dim = self.transformer.d_model
        # create conversion layer to connect the backbone to transformer
        self.conv = nn.Conv1d(self.backbone.num_channels*self.backbone.F, hidden_dim, 1)        
        self.pos_layer = PositionEmbeddingSinev2(**model_args.positional)

        # prediction heads, one extra class for predicting non-empty slots
#         self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_class = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(model_args.args.num_Q, hidden_dim)        


    def forward(self, waveform):
        """
        Input: (batch_size, data_length)"""
        x = self.spec_layer(waveform)
        x = x.transpose(1,2) # (B, T, n_mels)
        x = torch.log(x+epsilon) 
        x = x.unsqueeze(1) # (B, 1, T, n_mels)
        spec = x        
        
        x = self.backbone(x)
        backbone_feat = x
        # convert from 2048 to 256 feature planes for the transformer
        x = x.transpose(-1,-2).flatten(1,2)
        h = self.conv(x) # (B, C, T) (8, 256*8, 32) / (32, 128*14, 1001) / (32, 128*14, 62)
#         # construct positional encodings
        B, _, T = h.shape
        pos = self.pos_layer(h)
        hs = self.transformer(h, None, self.query_embed.weight, pos)[0]
        # (num_decoder_layers, B, Q, hidden_dim)      
        
        h = hs[-1] # (B, Q, hidden_dim)
        
        h = F.dropout(h, p=0.5)
#         h = F.relu_(self.fc1(h))
            
        aux_outputs = []
        for i in hs[:-1]:
            i = F.dropout(i, p=0.5)
#             i = F.relu_(self.fc1(i))
            aux_outputs.append({'pred_logits': self.linear_class(i).flatten(1)})
            
    
        return {'pred_logits': self.linear_class(h).flatten(1),
                'aux_outputs': aux_outputs,
                'spec': spec,
                'backbone_feat': backbone_feat}
    
    
class CombinedModel_Sv2_torch(nn.Module):
    """
    Combining Jointist CNN together with Transformer
    """
    def __init__(self, model_args, backbone, transformer, spec_args):

        super().__init__()
        self.num_classes = model_args.args.num_classes
        self.feature_weight = model_args.args.feature_weight
        self.spec_layer = transforms.MelSpectrogram(**spec_args.STFT)
        # decalre a backbone
        self.backbone = backbone
        # declare a transformer        
        self.transformer=transformer        
        hidden_dim = self.transformer.d_model
        # create conversion layer to connect the backbone to transformer
        self.conv = nn.Conv1d(self.backbone.num_channels*self.backbone.F, hidden_dim, 1)        
        self.pos_layer = PositionEmbeddingSinev2(**model_args.positional)

        # prediction heads, one extra class for predicting non-empty slots
#         self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_class = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(model_args.args.num_Q, hidden_dim)        


    def forward(self, waveform):
        """
        Input: (batch_size, data_length)"""
        x = self.spec_layer(waveform)
        x = x.transpose(1,2) # (B, T, n_mels)
        x = torch.log(x+epsilon) 
        x = x.unsqueeze(1) # (B, 1, T, n_mels)
        spec = x        
        
        x = self.backbone(x)
        backbone_feat = x        
        # convert from 2048 to 256 feature planes for the transformer
        x = x.transpose(-1,-2).flatten(1,2)
        h = self.conv(x) # (B, C, T) (8, 256*8, 32) / (32, 128*14, 1001) / (32, 128*14, 62)
#         # construct positional encodings
        B, _, T = h.shape
        pos_h = self.pos_layer(h)
        h = h.transpose(1,2)
        pos_h = pos_h.transpose(1,2) # (B, T, C) swap T and C for transformer
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        h = self.transformer(h+pos_h, query_embed)
        # (B, Q, hidden_dim)      
            
            
    
        return {'pred_logits': self.linear_class(h).flatten(1),
                'spec': spec,
                'backbone_feat': backbone_feat}    
    
    
class CombinedModel_Av2_Teacher(nn.Module):
    """
    Combining Jointist CNN together with Transformer
    """
    def __init__(self, model_args, backbone, transformer, spec_args):

        super().__init__()
        self.num_classes = model_args.args.num_classes
        self.feature_weight = model_args.args.feature_weight
        self.spec_layer = transforms.MelSpectrogram(**spec_args.STFT)
        # decalre a backbone
        self.backbone = backbone
        # declare a transformer        
        self.transformer = transformer
        hidden_dim = model_args.args.d_model
        # create conversion layer to connect the backbone to transformer
        self.conv = nn.Conv1d(self.backbone.num_channels*self.backbone.F, hidden_dim, 1)        
        self.pos_layer = PositionEmbeddingSinev2(**model_args.positional)

        # prediction heads, one extra class for predicting non-empty slots
#         self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_class = nn.Linear(hidden_dim, model_args.args.num_classes)
        self.query_embed = nn.Embedding(model_args.args.num_classes+1, hidden_dim) # + SOS token
        self.target_dropout = nn.Dropout2d(model_args.target_dropout)

    def forward(self, waveform, target, logit_scaler=None):
        """
        Input: (batch_size, data_length)"""
        x = self.spec_layer(waveform)
        x = x.transpose(1,2) # (B, T, n_mels)
        x = torch.log(x+epsilon) 
        x = x.unsqueeze(1) # (B, 1, T, n_mels)
        spec = x        
        
        x = self.backbone(x)
        backbone_feat = x
        # convert from 2048 to 256 feature planes for the transformer
        x = x.transpose(-1,-2).flatten(1,2)
        h = self.conv(x) # (B, C, T) (8, 256*8, 32) / (32, 128*14, 1001) / (32, 128*14, 62)
#         # construct positional encodings

        B, T = target.shape
        SOS_token = torch.zeros(B,1).fill_(self.num_classes).to(target.device).long() # (B, 1)
        target = torch.cat((SOS_token, target), dim=-1) # appending SOS token to the first timestep
        pos = self.pos_layer(h) # (B, C, T)
        h = (h+pos).transpose(1,2) # (B, T, C)        

        if self.training:
            tgt_mask = self.transformer.generate_square_subsequent_mask(T+1).to(h.device)
            tgt_embed = self.query_embed(target)
            tgt_embed = self.target_dropout(tgt_embed)
            output = self.transformer(h, tgt_embed, tgt_mask=tgt_mask)   
            output = output[:,:-1] # remove the last token 
            logits = self.linear_class(output) # (B, tgt_T, num_instruments)
            if logit_scaler!=None:
                logits = logits*logit_scaler
        else:
            input_seq = SOS_token
            tgt_embed = self.query_embed(input_seq) # (B, 1, C)
            stepwise_output = self.transformer(h, tgt_embed) # (B, 1, C)
            stepwise_output = self.linear_class(stepwise_output) # (B, 1, num_instruments)            
            input_seq = torch.cat((input_seq, stepwise_output.argmax(-1)), dim=1)
            output = stepwise_output # (B, 1, c)
            if logit_scaler!=None:    
                output = output*logit_scaler[:,:1] # get first element while keeping the shape
            for i in range(T-1): # don't need the last token
                tgt_embed = self.query_embed(input_seq) # (B, 1+i, f_dim)
                stepwise_output = self.transformer(h, tgt_embed) # (B, 1+i, f_dim)
                stepwise_output = self.linear_class(stepwise_output[:,-1:]) # (B, i+1, num_instruments)
                if logit_scaler!=None:
                    stepwise_output = stepwise_output*logit_scaler[:,i+1].unsqueeze(1)
                output = torch.cat((output, stepwise_output), dim=1) # (B, 1+i, num_instruments)  
                input_seq = torch.cat((input_seq, stepwise_output.argmax(-1)), dim=1) # (B, 1+i+1)  
            logits = output
#         memory = self.t_encoder(h) # no need mask for encoder, since no padding
        # (num_decoder_layers, B, Q, hidden_dim)
#         mask = torch.tril(torch.ones(T+1, T+1)).to(memory.device) # Lower triangular matrix for masking the target, +1 for SOS token

        
#         h = self.t_decoder(tgt_embed, memory, tgt_mask=mask) # (B, tgt_T, hidden_dim)
#         h = F.relu_(self.fc1(h))
            
        return {'logits': logits,
                'spec': spec,
                'backbone_feat': backbone_feat}
    
    
class CombinedModel_Av2(nn.Module):
    """
    Combining Jointist CNN together with Transformer
    """
    def __init__(self, model_args, backbone, transformer, spec_args):

        super().__init__()
        self.num_classes = model_args.args.num_classes
        self.feature_weight = model_args.args.feature_weight
        self.spec_layer = transforms.MelSpectrogram(**spec_args.STFT)
        # decalre a backbone
        self.backbone = backbone
        # declare a transformer        
        self.transformer = transformer
        hidden_dim = model_args.args.d_model
        # create conversion layer to connect the backbone to transformer
        self.conv = nn.Conv1d(self.backbone.num_channels*self.backbone.F, hidden_dim, 1)        
        self.pos_layer = PositionEmbeddingSinev2(**model_args.positional)

        # prediction heads, one extra class for predicting non-empty slots
#         self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_class = nn.Linear(hidden_dim, model_args.args.num_classes)
        self.query_embed = nn.Embedding(model_args.args.num_classes+1, hidden_dim) # + SOS token

    def forward(self, waveform, target):
        """
        Input: (batch_size, data_length)"""
        x = self.spec_layer(waveform)
        x = x.transpose(1,2) # (B, T, n_mels)
        x = torch.log(x+epsilon) 
        x = x.unsqueeze(1) # (B, 1, T, n_mels)
        spec = x        
        
        x = self.backbone(x)
        backbone_feat = x
        # convert from 2048 to 256 feature planes for the transformer
        x = x.transpose(-1,-2).flatten(1,2)
        h = self.conv(x) # (B, C, T) (8, 256*8, 32) / (32, 128*14, 1001) / (32, 128*14, 62)
#         # construct positional encodings

        B, T = target.shape
        SOS_token = torch.zeros(B,1).fill_(self.num_classes).to(target.device).long() # (B, 1)
        target = torch.cat((SOS_token, target), dim=-1) # appending SOS token to the first timestep
        pos = self.pos_layer(h) # (B, C, T)
        h = (h+pos).transpose(1,2) # (B, T, C)        
     
        input_seq = SOS_token
        tgt_embed = self.query_embed(input_seq) # (B, 1, C)
        stepwise_output = self.transformer(h, tgt_embed) # (B, 1, C)
        stepwise_output = self.linear_class(stepwise_output) # (B, tgt_T, num_instruments)            
        input_seq = torch.cat((input_seq, stepwise_output.argmax(-1)), dim=1)
        output = stepwise_output # (B, 1, c)
        for i in range(T-1): # don't need the last token
            tgt_embed = self.query_embed(input_seq) # (B, 1+i, f_dim)
            stepwise_output = self.transformer(h, tgt_embed) # (B, 1+i, f_dim)
            stepwise_output = self.linear_class(stepwise_output) # (B, i+1, num_instruments)  
            output = torch.cat((output, stepwise_output[:,-1:]), dim=1) # (B, 1+i, num_instruments)  
            input_seq = torch.cat((input_seq, stepwise_output[:,-1:].argmax(-1)), dim=1) # (B, 1+i+1, num_instruments)  
        logits = output
#         memory = self.t_encoder(h) # no need mask for encoder, since no padding
        # (num_decoder_layers, B, Q, hidden_dim)
#         mask = torch.tril(torch.ones(T+1, T+1)).to(memory.device) # Lower triangular matrix for masking the target, +1 for SOS token

        
#         h = self.t_decoder(tgt_embed, memory, tgt_mask=mask) # (B, tgt_T, hidden_dim)
#         h = F.relu_(self.fc1(h))
            
        return {'pred_logits': logits,
                'spec': spec,
                'backbone_feat': backbone_feat}       
        

class CombinedModel_Linear(nn.Module):
    """
    Combining Jointist CNN together with Transformer
    """
    def __init__(self, model_args, backbone, linear, spec_args):

        super().__init__()
        self.num_classes = model_args.args.num_classes
        self.feature_weight = model_args.args.feature_weight
        self.spec_layer = transforms.MelSpectrogram(**spec_args.STFT)
        # decalre a backbone
        self.backbone = backbone
        # declare a transformer
        hidden_dim = linear.out_features
        # create conversion layer to connect the backbone to transformer
        self.conv = nn.Conv2d(self.backbone.num_channels, hidden_dim, 1)        

        self.linear = linear
        # prediction heads, one extra class for predicting non-empty slots
        self.linear_class = nn.Linear(hidden_dim, self.num_classes)   


    def forward(self, waveform):
        """
        Input: (batch_size, data_length)"""
        x = self.spec_layer(waveform)
        x = x.transpose(1,2) # (B, T, n_mels)
        x = torch.log(x+epsilon) 
        x = x.unsqueeze(1) # (B, 1, T, n_mels)
        spec = x        
        
        x = self.backbone(x)
        backbone_feat = x
        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x) # (B, hidde) (8, 256, 32, 8) or (8,256,15,3)
#         # construct positional encodings
        B, _, H, W = h.shape
        h = self.linear(h.flatten(1)) # (8,256)  # This is actually a linear layer
        pred_logits = self.linear_class(h)
    
        return {'pred_logits': pred_logits,
                'spec': spec,
                'backbone_feat': backbone_feat}
    

class Original(nn.Module):
    def __init__(self, model_args, backbone, spec_args):

        super().__init__()
        self.num_classes = model_args.args.num_classes
        # Spectrogram extractor
        self.spec_layer = transforms.MelSpectrogram(**spec_args.STFT)
        self.backbone = backbone

        self.fc1 = nn.Linear(self.backbone.num_channels, 2048, bias=True)
        
        # In slack we need self.num_classes-1, since empty is one of the classes
        self.fc_audioset = nn.Linear(2048, self.num_classes-1, bias=True)        


        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, waveform):
        """
        Input: (batch_size, data_length)"""

        x = self.spec_layer(waveform)
        x = x.transpose(1,2) # (B, T, n_mels)
        x = torch.log(x+epsilon) 
        x = x.unsqueeze(1) # (B, 1, T, n_mels)
        spec = x        
        
        x = self.backbone(x)
        backbone_feat = x

        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5)
        x = F.relu_(self.fc1(x))
        clipwise_output = self.fc_audioset(x)

        return {'pred_logits': clipwise_output,
                'backbone_feat': backbone_feat,
                'spec': spec}    
                
        
class CombinedModel_CLS(nn.Module):
    """
    Combining Jointist CNN together with Transformer
    """
    def __init__(self, model_args, backbone, encoder, spec_args):

        super().__init__()
        self.num_classes = model_args.args.num_classes
        self.feature_weight = model_args.args.feature_weight
        self.spec_layer = transforms.MelSpectrogram(**spec_args.STFT)
        # decalre a backbone
        self.backbone = backbone
        # declare a transformer        
        self.encoder=encoder        
        hidden_dim = self.encoder.hidden_size
        # create conversion layer to connect the backbone to transformer
        self.conv = nn.Conv2d(self.backbone.num_channels, hidden_dim, 1)        
        self.pooler = BertPooler(self.encoder.bert_config)

        self.dropout = nn.Dropout(0.5)
        self.linear_class = nn.Linear(hidden_dim, self.num_classes)
        
        # For adding CLS tokens
        self.vec_cls = self.get_cls(hidden_dim)

    def get_cls(self, channel):
        np.random.seed(0)
        single_cls = torch.Tensor(np.random.random((1, channel)))
        vec_cls = torch.cat([single_cls for _ in range(64)], dim=0)
        vec_cls = vec_cls.unsqueeze(1)
        return vec_cls

    def append_cls(self, x):
        batch, _, _ = x.size()
        part_vec_cls = self.vec_cls[:batch].clone()
        part_vec_cls = part_vec_cls.to(x.device)
        return torch.cat([part_vec_cls, x], dim=1)        


    def forward(self, waveform):
        """
        Input: (batch_size, data_length)"""
        x = self.spec_layer(waveform)
        x = x.transpose(1,2) # (B, T, n_mels)
        x = torch.log(x+epsilon) 
        x = x.unsqueeze(1) # (B, 1, T, n_mels)
        spec = x        
        
        x = self.backbone(x)
        backbone_feat = x
        # convert from 2048 to 256 feature planes for the transformer
        x = self.conv(x).flatten(2) # (B, hidden, T*F) (8, 256, 32*8)/(8, 128, 32*8)
        
        x = x.permute(0,2,1) # (B, T*F, hidden)
        x = self.append_cls(x) # (B, 1+T*F, hidden)

        x = self.encoder(x) 
        x = x[-1] # (B, 1+T*F, hidden)
        x = self.pooler(x) # (B, hidden) using the CLS token

        # Dense
        x = self.dropout(x)
        pred_logits = self.linear_class(x) # (B, num_classes)
            
    
        return {'pred_logits': pred_logits,
                'spec': spec,
                'backbone_feat': backbone_feat}
    
class CombinedModel_CLSv2(nn.Module):
    """
    Combining Jointist CNN together with Transformer
    """
    def __init__(self, model_args, backbone, encoder, spec_args):

        super().__init__()
        self.num_classes = model_args.args.num_classes
        self.feature_weight = model_args.args.feature_weight
        self.spec_layer = transforms.MelSpectrogram(**spec_args.STFT)
        # decalre a backbone
        self.backbone = backbone
        # declare a transformer        
        self.encoder=encoder        
#         hidden_dim = self.encoder.hidden_size
        hidden_dim = self.encoder.hidden_size
        # create conversion layer to connect the backbone to transformer
        # TODO: prevent hardcoding the 256 channels
        self.conv = nn.Conv1d(self.backbone.num_channels*self.backbone.F, hidden_dim, 1)        
        self.pooler = BertPooler(self.encoder.bert_config)

        self.dropout = nn.Dropout(0.5)
        self.linear_class = nn.Linear(hidden_dim, self.num_classes)
        
        # For adding CLS tokens
        self.vec_cls = self.get_cls(hidden_dim)

    def get_cls(self, channel):
        np.random.seed(0)
        single_cls = torch.Tensor(np.random.random((1, channel)))
        vec_cls = torch.cat([single_cls for _ in range(64)], dim=0)
        vec_cls = vec_cls.unsqueeze(1)
        return vec_cls

    def append_cls(self, x):
        batch, _, _ = x.size()
        part_vec_cls = self.vec_cls[:batch].clone()
        part_vec_cls = part_vec_cls.to(x.device)
        return torch.cat([part_vec_cls, x], dim=1)        


    def forward(self, waveform):
        """
        Input: (batch_size, data_length)"""
        x = self.spec_layer(waveform)
        x = x.transpose(1,2) # it is required for batchnorm in the backbone # (B, T, n_mels)
        x = torch.log(x+epsilon) 
        x = x.unsqueeze(1) # (B, 1, T, n_mels)
        spec = x       
        
        x = self.backbone(x)
        backbone_feat = x
        x = x.transpose(2,3) # (B, 1, n_mels, T)     
        # convert from 2048 to 256 feature planes for the transformer
        x = x.flatten(1,2) # combining C and F dimensions (8, 256*3, 15) 
        x = self.conv(x) # (B, hidden, T*F) (8, 256, 3, 15)          
        
        x = x.permute(0,2,1) # (B, T, C*F)
        x = self.append_cls(x) # (B, 1+T, C*F)

        x = self.encoder(x)     
        x = x[-1] # (B, 1+T, C*F)
        x = self.pooler(x) # (B, hidden) using the CLS token

        # Dense
        x = self.dropout(x)
        pred_logits = self.linear_class(x) # (B, num_classes)
            
    
        return {'pred_logits': pred_logits,
                'spec': spec,
                'backbone_feat': backbone_feat}    
    
    
    
class CombinedModel_NewCLSv2(nn.Module):
    """
    Combining Jointist CNN together with Transformer
    """
    def __init__(self, model_args, backbone, encoder, spec_args):

        super().__init__()
        self.num_classes = model_args.args.num_classes
        self.feature_weight = model_args.args.feature_weight
        self.spec_layer = transforms.MelSpectrogram(**spec_args.STFT)
        # decalre a backbone
        self.backbone = backbone
        # declare a transformer        
        self.encoder=encoder        
#         hidden_dim = self.encoder.hidden_size
        hidden_dim = 256
        # create conversion layer to connect the backbone to transformer
        # TODO: prevent hardcoding the 256 channels
        self.conv = nn.Conv1d(self.backbone.num_channels*self.backbone.F, hidden_dim, 1)        

        self.dropout = nn.Dropout(0.5)
        self.linear_class = nn.Linear(hidden_dim, self.num_classes-1)


    def forward(self, waveform):
        """
        Input: (batch_size, data_length)"""
        x = self.spec_layer(waveform)
        x = x.transpose(1,2) # it is required for batchnorm in the backbone # (B, T, n_mels)
        x = torch.log(x+epsilon) 
        x = x.unsqueeze(1) # (B, 1, T, n_mels)
        spec = x       
        
        x = self.backbone(x)
        backbone_feat = x
        
        x = x.transpose(-1,-2).flatten(1,2)
        x = self.conv(x) # (B, C, T) (8, 256*3, 15) / (32, 128*14, 1001) / (32, 128*14, 62)
        x = x.transpose(1,2) # (B, T, C)
        x = self.encoder(x) # CLS token output (B, C)

        # Dense
        x = self.dropout(x)
        pred_logits = self.linear_class(x) # (B, num_classes)
            
    
        return {'pred_logits': pred_logits,
                'spec': spec,
                'backbone_feat': backbone_feat}    
                            
                        
        
class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output            