from jiwer import cer,wer
import torch
import torch.nn as nn
from model import Acoustic_Phonetic_Linguistic
import numpy as np
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
from char_embedding import tensor_to_text
# from model import Net
from infer import phonetic_embedding
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torch.nn as nn
import torch.nn.functional as F
from phonetic_encoder import Phonetic_encoder
from acoustic_encoder import Acoustic_encoder
from linguistic_encoder import Linguistic_encoder
from char_embedding import text_to_tensor
from help import wav_norm, Atention
import numpy as np
# from help import
from model import Acoustic_Phonetic_Linguistic
from dataloader import MDD_Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.utils.rnn import pack_sequence
tokenizer = Wav2Vec2Processor.from_pretrained("pretrained_finetuned")
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
import pandas as pd
data = pd.read_csv("/home/tuht/train_wav2vec/test.csv")
# sample = data.shape[0]
f = open("./cer.txt", 'a')
fw = open("./wer.txt", 'a')
# print(data['sentence'])

net = Acoustic_Phonetic_Linguistic()

net = torch.load('/home/tuht/PL/MDD_Checkpoint/checkpoint_AdamW_16head_PL.pth10')
net.to('cuda')
net.eval()
acoustic_dir = '/home/tuht/train_wav2vec/acoustic/'
phonetic_dir = '/home/tuht/train_wav2vec/phonetic/'
pitch_dir = '/home/tuht/train_wav2vec/pitch/'
for i in range(len(data)):
    print(i)
    path = data['Path'][i]
    can = data['Canonical'][i]
    transcript = data['Transcript'][i]
    p = path
    p = p + str(".npy")
    p = np.load(phonetic_dir + str(p))
    phonetic = torch.tensor(p)
    # acoustic = path + str(".wav")
    # path = acoustic_dir + acoustic
    
    # acoustic = acoustic
    acoustic = torch.tensor(0)
    phonetic = phonetic

    linguistic = text_to_tensor(can)
    linguistic = torch.tensor(linguistic)

    acoustic = acoustic.to('cuda')
    phonetic = phonetic.to('cuda')
    linguistic = linguistic.to('cuda')
    acoustic = acoustic.unsqueeze(0)
    phonetic = phonetic.unsqueeze(0)
    linguistic = linguistic.unsqueeze(0)
    outputs = net(acoustic,phonetic,linguistic)
    outputs = outputs.unsqueeze(0)
    x = F.log_softmax(outputs,dim=2)
    x = x.unsqueeze(1)
    labels = ['ắ', 'ồ', 'z', 'ứ', 'ỡ', 'ì', 'x', 'ặ', 'u', 'ẹ', 'd', 'ỵ', 'r', 'p', 't', 'ỳ', 'ẩ', 'f', 'ó', 'á', 'v', 'ã', 'i', 'ư', 'ở', 'ễ', 'ụ', 'ú', 'ũ', ' ', 'ă', 'é', 'ằ', 'a', 'ấ', 'ờ', 'ữ', 'ớ', 'n', 'ý', 's', 'h', 'ơ', 'ị', 'l', 'c', 'k', 'ỷ', 'ỗ', 'ế', 'ẻ', 'ợ', 'ẫ', 'í', 'ỏ', 'ủ', 'g', 'q', 'j', 'ò', 'ỹ', 'ự', 'ô', 'b', 'y', 'ĩ', 'ỉ', 'ẵ', 'ầ', 'ê', 'ộ', 'ậ', 'm', 'ń', 'o', 'ọ', 'đ', 'ẽ', 'ử', 'à', 'è', 'e', 'ẳ', 'ổ', 'ù', 'w', 'ả', 'ạ', 'â', 'ệ', 'ề', 'õ', 'ố', 'ể', 'ừ',]
    x = x.squeeze(1)
    x = x.detach().cpu().numpy()
    decoder = build_ctcdecoder(
        labels = labels,
        
    )
    x = x.squeeze(0)
    ground_truth = [str(transcript)]
    hypothesis = [str(decoder.decode(x))]
    error = cer(ground_truth, hypothesis)
    f.write(str(error) + "\n") 
    error = wer(ground_truth, hypothesis)
    fw.write(str(error) + "\n") 
    # print(error)
    

