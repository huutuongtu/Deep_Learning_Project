from cmath import acos
import torch
from torch.utils.data import Dataset
# import matplotlib.pyplot as plt
import os
import pandas as pd
# import numpy as np
# from torch.utils.data import DataLoader
from infer import phonetic_embedding
from help import Atention, wav_norm        
from char_embedding import tensor_to_text,text_to_tensor
import librosa
import numpy as np

data = pd.read_csv("/home/tuht/train_wav2vec/vi/vi/validated.csv", sep='\t')
sample = data.shape[0]
cols = ['path', 'sentence']

class MDD_Dataset(Dataset):

    def __init__(self):
        acoustic_canonical = data
        self.n_samples = sample
        A = acoustic_canonical['path']
        C = acoustic_canonical['sentence']
        B = acoustic_canonical['sentence'] #output
        

        self.A_data = A 
        self.C_data = C
        self.y_data = B 

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        p = self.A_data[index]
        p = p.split("/")[7]
        p = p.split(".")[0]
        base_dir = '/home/tuht/train_wav2vec/phonetic/'
        p = p + str(".npy")
        p = np.load(base_dir + str(p))
        phonetic = torch.tensor(p)
        # print(self.A_data[index])
        acoustic = wav_norm(self.A_data[index])
        acoustic = acoustic.T
        acoustic = torch.tensor(acoustic)
        acoustic = acoustic.unsqueeze(0)
        linguistic = text_to_tensor(self.C_data[index])
        linguistic = torch.tensor(linguistic)
        label = text_to_tensor(self.y_data[index])
        label = torch.tensor(label)
        return acoustic.squeeze(0), phonetic, linguistic, label

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


