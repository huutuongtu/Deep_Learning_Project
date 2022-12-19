from unicodedata import bidirectional
import torch.nn as nn
import torch.nn.functional as F
import torch        


class Linguistic_encoder(nn.Module):
    def __init__(self):
        super().__init__()

        
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.fc1 = nn.Linear(64, 1536)
        self.fc2 = nn.Linear(1,64) 
        self.bilstm = nn.LSTM(input_size= 64, hidden_size = 64,bidirectional = True)
        self.fc3 = nn.Linear(64,1536)
        self.bn = nn.BatchNorm1d(1536)
        # self.Embedding_Layer = nn.Embedding(200, 64)

        


    def forward(self, x):
        # x = self.fc2(x)
        x = torch.t(x)  #(len(canonical) x 1)
        x = torch.tensor(x, dtype=torch.float)
        # print(x.shape)
        x = self.fc2(x)
        # print(x.shape)
        o, (h_n, c_n) = self.bilstm(x)
        # print(o.shape)
        # print("Tu")
        # print(h_n.shape)
        # print(c_n)
        y = self.fc1(h_n)
        x = self.fc3(h_n)

        return x,y


        