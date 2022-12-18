from unicodedata import bidirectional
import torch.nn as nn
import torch.nn.functional as F
import torch
from help import wav_norm

class CNN_Stack1(nn.Module):
    def __init__(self):
        super().__init__()


        self.bn = nn.BatchNorm1d(81)
        self.Conv2d = nn.Conv2d(1,1,3,1,1)
        self.reLU = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.2)
        # self.bilstm = nn.LSTM(input_size= 97, hidden_size = 97,bidirectional = True)


    def forward(self, x):
        x = (self.Conv2d(x))
        x = x.squeeze(0)
        x = torch.t(x)
        x = x.unsqueeze(0)
        x = self.bn(x)
        x = self.reLU(x)
        x = self.drop_out(x)
        
        return x

class CNN_Stack(nn.Module):
    def __init__(self):
        super().__init__()


        self.bn = nn.BatchNorm1d(81)
        self.Conv2d = nn.Conv2d(1,1,3,1,1)
        self.reLU = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.2)
        # self.bilstm = nn.LSTM(input_size= 97, hidden_size = 97,bidirectional = True)


    def forward(self, x):
        x = (self.Conv2d(x))
        x = self.bn(x)
        x = self.reLU(x)
        x = self.drop_out(x)
        
        return x

class RNN_Stack_1(nn.Module):
    def __init__(self):
        super().__init__()
 
        self.reLU = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.2)
        self.bilstm = nn.LSTM(input_size= 81, hidden_size = 384,bidirectional = True)
        self.bn = nn.BatchNorm1d(768)

    def forward(self, x):
        x = x.squeeze(0)
        x = torch.t(x)
        x = x.unsqueeze(1)
        x = self.bilstm(x)
        x = self.drop_out(x[0])
        x = x.squeeze(1)
        x = torch.t(x)
        x = x.unsqueeze(0)
        x = self.bn(x)
        return x.unsqueeze(0)


class RNN_Stack(nn.Module):
    def __init__(self):
        super().__init__()
 
        self.reLU = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.2)
        self.bilstm = nn.LSTM(input_size= 768, hidden_size = 384,bidirectional = True)
        self.bn = nn.BatchNorm1d(768)

    def forward(self, x):
        x = x.squeeze(0).squeeze(0)
        x = torch.t(x)
        x = x.unsqueeze(1)
        x = self.bilstm(x)
        x = self.drop_out(x[0])
        x = x.squeeze(1)
        x = torch.t(x)
        x = x.unsqueeze(0)
        
        # print(x.shape)
        x = self.bn(x)
        return x.unsqueeze(0)
        
        return x.unsqueeze(0)
        
        


class Acoustic_encoder(nn.Module):
    def __init__(self):
        super().__init__()


        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.CNN1 = CNN_Stack1()
        self.CNN = CNN_Stack()
        self.RNN = RNN_Stack()
        self.RNN1 = RNN_Stack_1()

        


    def forward(self, x):
        # x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.CNN1(x)
        x = self.CNN(x)
        # print(x.shape)
        x = self.RNN1(x)
        x = self.RNN(x)
        # x = self.RNN(x)
        # x = self.RNN(x)
        return x
        
