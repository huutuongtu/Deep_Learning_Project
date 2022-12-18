from unicodedata import bidirectional
import torch.nn as nn
import torch.nn.functional as F
import torch
from linguistic_encoder import Linguistic_encoder
from acoustic_encoder import Acoustic_encoder
from phonetic_encoder import Phonetic_encoder
from help import Atention, wav_norm        
from char_embedding import tensor_to_text,text_to_tensor
import numpy as np
import torch
import torch.nn as nn
from infer import phonetic_embedding

class Acoustic_Phonetic_Linguistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.Acoustic_encoder = Acoustic_encoder()
        self.Phonetic_encoder = Phonetic_encoder()
        self.Linguistic_encoder = Linguistic_encoder()
        self.text_to_tensor = text_to_tensor
        self.tensor_to_text = tensor_to_text
        self.phonetic_embedding = phonetic_embedding
        self.Attention = Atention
        self.wav_norm = wav_norm   
        self.fc1 = nn.Linear(3072,96, bias = True)  
        self.fc2 = nn.Linear(768, 1536)   
        self.multihead_attn = nn.MultiheadAttention(1536, 16, batch_first=True)
        


    def forward(self, acoustic, phonetic, linguistic):
  
        phonetic = self.Phonetic_encoder(phonetic) #batch x time x 768
        # acoustic = self.Acoustic_encoder(acoustic) #batch x time x 768
        linguistic = self.Linguistic_encoder(linguistic) # shape [0]: 1536 x len(canon)
        Hv = linguistic[0] 
        Hk = linguistic[1] 
        phonetic = self.fc2(phonetic)
        # acoustic = acoustic.squeeze(0).squeeze(0)
        # acoustic = torch.t(acoustic)
        # acoustic = acoustic.unsqueeze(0)
        # Hq = (torch.cat((acoustic,phonetic),2))
        Hq = phonetic
        Hk = Hk.unsqueeze(0)
        Hv = Hv.unsqueeze(0)
        attn_output, attn_output_weights = self.multihead_attn(Hq, Hk, Hv)
        c = attn_output
        # print("A")
        # print(o,s)
        # print(a)
        before_Linear = torch.cat((c,Hq), 2)
        # print(before_Linear.shape)
        output = self.fc1(before_Linear)
        # print(output.shape)
        return output.squeeze(0)
        
        