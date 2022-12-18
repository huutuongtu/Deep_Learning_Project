from model import Acoustic_Phonetic_Linguistic
import torch
from cmath import acos
import torch
# from torch.utils.data import Dataset
# import matplotlib.pyplot as plt
# import os
# import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from infer import phonetic_embedding
from help import Atention, wav_norm        
from char_embedding import tensor_to_text,text_to_tensor
import librosa
import torch.nn.functional as F
import torch.nn as nn
from pyctcdecode import build_ctcdecoder
# from jiwer import cer
import torch
import torch.nn as nn
from model import Acoustic_Phonetic_Linguistic
import numpy as np
from char_embedding import tensor_to_text
# import matplotlib.pyplot as plt

x = np.load("/home/tuht/train_wav2vec/output.npy")


x = torch.tensor(x)



x = F.log_softmax(x,dim=1)

x = x.unsqueeze(1)

print([x.shape[0]])
ctc_loss = nn.CTCLoss(blank = 95)
can = 'xin chào'
can = text_to_tensor(can)

print(can)

target = torch.tensor(can)
target = target.unsqueeze(0)
input_length = torch.tensor([x.shape[0]])
output_length = torch.tensor([len(can)])
print(ctc_loss(x, target, input_length, output_length))
x = x.squeeze(1)


labels = ['ắ', 'ồ', 'z', 'ứ', 'ỡ', 'ì', 'x', 'ặ', 'u', 'ẹ', 'd', 'ỵ', 'r', 'p', 't', 'ỳ', 'ẩ', 'f', 'ó', 'á', 'v', 'ã', 'i', 'ư', 'ở', 'ễ', 'ụ', 'ú', 'ũ', ' ', 'ă', 'é', 'ằ', 'a', 'ấ', 'ờ', 'ữ', 'ớ', 'n', 'ý', 's', 'h', 'ơ', 'ị', 'l', 'c', 'k', 'ỷ', 'ỗ', 'ế', 'ẻ', 'ợ', 'ẫ', 'í', 'ỏ', 'ủ', 'g', 'q', 'j', 'ò', 'ỹ', 'ự', 'ô', 'b', 'y', 'ĩ', 'ỉ', 'ẵ', 'ầ', 'ê', 'ộ', 'ậ', 'm', 'ń', 'o', 'ọ', 'đ', 'ẽ', 'ử', 'à', 'è', 'e', 'ẳ', 'ổ', 'ù', 'w', 'ả', 'ạ', 'â', 'ệ', 'ề', 'õ', 'ố', 'ể', 'ừ',]
x = x.detach().cpu().numpy()
decoder = build_ctcdecoder(
    labels = labels,
    
)
# x = x.squeeze(0)
print(len(decoder.decode(x)))
print((decoder.decode(x)))


