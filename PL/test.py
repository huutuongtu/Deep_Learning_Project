from model import Acoustic_Phonetic_Linguistic
import torch
from cmath import acos
import torch
from torch.utils.data import Dataset
# import matplotlib.pyplot as plt
# import os
# import pandas as pd
import numpy as np
# from torch.utils.data import DataLoader
from infer import phonetic_embedding
from help import Atention, wav_norm        
from char_embedding import tensor_to_text,text_to_tensor
import librosa
net = Acoustic_Phonetic_Linguistic()
# net.eval()
net = net.to('cpu')
net = torch.load('/home/tuht/train_wav2vec/MDD_Checkpoint/checkpoint_AdamW_16head_PL.pth')

net.eval()



path = '/home/tuht/train_wav2vec/common_voice_vi_21833214.wav'
can = 'dứt lời trinh vội bỏ ra ngoài quên lấy cả tiền'
phonetic = phonetic_embedding(path)
acoustic = wav_norm(path)
acoustic = acoustic.T
acoustic = torch.tensor(acoustic)
acoustic = acoustic.unsqueeze(0)
phonetic = phonetic
linguistic = text_to_tensor(can)
linguistic = torch.tensor(linguistic)
linguistic = linguistic.unsqueeze(1)
linguistic = torch.t(linguistic)
acoustic = acoustic.to('cuda')
phonetic = phonetic.to('cuda')
linguistic = linguistic.to('cuda')
# acoustic = acoustic.
# print(acoustic.shape)
# print(phonetic.shape)
# print(linguistic.shape)
outputs = net(acoustic,phonetic,linguistic)
predicted_ids = torch.argmax(outputs, dim=1)
print(predicted_ids)
outputs = outputs.cpu().detach().numpy()
np.save('output.npy', outputs)


