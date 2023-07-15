import math
from math import floor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torchaudio import transforms

def init_layer(layer):
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)

def init_bn(bn):
    bn.weight.data.fill_(1.)

class Attention(nn.Module):
    def __init__(self, n_in, n_out):
        super(Attention, self).__init__()

        self.att = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.cla = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.att,)
        init_layer(self.cla)

    def forward(self, x):
        """input: (samples_num, freq_bins, time_steps, 1)
        """

        att = self.att(x)
        att = torch.sigmoid(att)

        cla = self.cla(x)
        cla = torch.sigmoid(cla)

        att = att[:, :, :, 0]   # (samples_num, classes_num, time_steps)
        cla = cla[:, :, :, 0]   # (samples_num, classes_num, time_steps)

        epsilon = 1e-7
        att = torch.clamp(att, epsilon, 1. - epsilon)

        norm_att = att / torch.sum(att, dim=2)[:, :, None]
     
        x = torch.sum(norm_att * cla, dim=2)
        
        x = F.hardtanh(x, 0., 1.)
        return x

class EmbeddingLayers(nn.Module):

    def __init__(self, freq_bins, emb_layers, hidden_units, drop_rate):
        super(EmbeddingLayers, self).__init__()

        self.freq_bins = freq_bins
        self.hidden_units = hidden_units
        self.drop_rate = drop_rate

        self.conv1x1 = nn.ModuleList()
        self.batchnorm = nn.ModuleList()

        for i in range(emb_layers):
            in_channels = freq_bins if i == 0 else hidden_units
            conv = nn.Conv2d(
                in_channels=in_channels, out_channels=hidden_units,
                kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
            self.conv1x1.append(conv)
            self.batchnorm.append(nn.BatchNorm2d(in_channels))

        # Append last batch-norm layer
        self.batchnorm.append(nn.BatchNorm2d(hidden_units))

        self.init_weights()

    def init_weights(self):

        for conv in self.conv1x1:
            init_layer(conv)

        for bn in self.batchnorm:
            init_bn(bn)

    def forward(self, x, return_layers=False):
        """input: (samples_num, time_steps, freq_bins)
        """

        drop_rate = self.drop_rate

        # (samples_num, freq_bins, time_steps)

        # Add an extra dimension for using Conv2d
        # (samples_num, freq_bins, time_steps, 1)
        
        x = x[:, :, :, None].contiguous()

        out = self.batchnorm[0](x)
        residual = x
        all_outs = [out]

        for i in range(len(self.conv1x1)):
            out = F.dropout(F.relu(self.batchnorm[i+1](self.conv1x1[i](out))),
                            p=drop_rate,
                            training=self.training)
            all_outs.append(out)
        out = out + residual
        if return_layers is False:
            # (samples_num, hidden_units, time_steps, 1)
            return out

        else:
            return all_outs

class DecisionLevelSingleAttention(nn.Module):

    def __init__(self, freq_bins, classes_num, emb_layers, hidden_units, drop_rate, spec_args):

        super(DecisionLevelSingleAttention, self).__init__()

        self.num_classes=classes_num-1 #remove the empty class
        self.spec_layer = transforms.MelSpectrogram(**spec_args.STFT)
        
        self.emb = EmbeddingLayers(
            freq_bins=freq_bins,
            emb_layers=emb_layers,
            hidden_units=hidden_units,
            drop_rate=drop_rate)

        self.attention = Attention(
            n_in=hidden_units,
            n_out=self.num_classes)
            
        self.param_count = count_parameters(self)
        print(self.param_count)

    def init_weights(self):
        pass

    def forward(self, input):
        """input: (samples_num, freq_bins, time_steps, 1)
        """
        spec = self.spec_layer(input)

        # (samples_num, hidden_units, time_steps, 1)
        b1 = self.emb(spec)

        # (samples_num, classes_num, time_steps, 1)
        output = self.attention(b1)

        return {"pred": output,
                "spec": spec}

def count_parameters(model):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            # if param.dim() > 1:
            #     print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            # else:
            #     print(name, ':', num_param)
            total_param += num_param
    # print('Total Parameters: {}'.format(total_param))
    return total_param
    
class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.FC = nn.Sequential(
            nn.Linear(1280, 512),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            # nn.Linear(512, 512),
            # nn.Dropout(0.5),
            # nn.LeakyReLU(),
            # nn.Linear(512, 512),
            # nn.Dropout(0.5),
            # nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(128, 20)
        )
        self.param_count = count_parameters(self)
        print(self.param_count)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self, X):
        X = X.view(X.shape[0], -1)
        out = torch.sigmoid(self.FC(X))
        return out
     
# I used this     
class FC_T(nn.Module):
    def __init__(self):
        super(FC_T, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(128, 128),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            # nn.Linear(512, 512),
            # nn.Dropout(0.5),
            # nn.LeakyReLU(),
            # nn.Linear(512, 512),
            # nn.Dropout(0.5),
            # nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
        )
        
        self.classify = nn.Linear(128, 20)
        self.param_count = count_parameters(self)
        print(self.param_count)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self, X):
        out = self.embed(X)
        out = (out+X).mean(1)
        out = torch.sigmoid(self.classify(out))
        return out
        
# Same model as above with max pooling instead of mean pooling
# class FC_T(nn.Module):
    # def __init__(self):
        # super(FC_T, self).__init__()
        # self.embed = nn.Sequential(
            # nn.Linear(128, 128),
            # nn.Dropout(0.6),
            # nn.LeakyReLU(),
            # nn.Linear(128, 128),
            # nn.Dropout(0.6),
            # nn.LeakyReLU(),
            # # nn.Linear(512, 512),
            # # nn.Dropout(0.5),
            # # nn.LeakyReLU(),
            # # nn.Linear(512, 512),
            # # nn.Dropout(0.5),
            # # nn.LeakyReLU(),
            # nn.Linear(128, 128),
            # nn.Dropout(0.6),
            # nn.LeakyReLU(),
        # )
        
        # self.classify = nn.Linear(128, 20)
        # self.param_count = count_parameters(self)
        # print(self.param_count)
        
        # for m in self.modules():
            # if isinstance(m, nn.Linear):
                # init.xavier_normal_(m.weight.data)
                # if m.bias is not None:
                    # m.bias.data.zero_()
    # def forward(self, X):
        # out = self.embed(X)
        # out,_ = (out+X).max(1)
        # out = torch.sigmoid(self.classify(out))
        # return out


class BaselineRNN_2(nn.Module):
    def __init__(self):
        super(BaselineRNN_2, self).__init__()
        self.rnn = nn.GRU(128, 64, num_layers=3, bidirectional=True, dropout=0.5, batch_first=True)
        self.FC = nn.Linear(128, 20)
        self.param_count = count_parameters(self)
        print(self.param_count)
    def forward(self, X):
        out, _ = self.rnn(X)
        out = torch.sigmoid(self.FC(out[:,-1,:]))
        return out    