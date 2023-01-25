from tkinter import N
import speechpy
import librosa
# import pandas as pd
import numpy as np
from char_embedding import text_to_tensor,tensor_to_text
import torch

def wav_norm(path):
    y, sr = librosa.load(path)
    t = librosa.get_duration(filename = path)
    data = librosa.feature.mfcc(y, sr=22050, hop_length=441, n_mfcc = 80)
    energy = librosa.feature.rms(y=y, hop_length = 441)
    data = np.concatenate((data,energy),axis=0) #acoustic data
    if (t/0.02-(t/0.02)//1 <=0.247) or (t/0.02-(t/0.02)//1 >0.999):
        data = np.delete(data,obj = [len(data[0])-1,len(data[0])-2],axis = 1)
    else:
        data = np.delete(data,obj = len(data[0])-1, axis = 1) 
    return data

#output, answ: list, 

def Atention(Hq, Hk):
    w = Hq.shape[0] #time
    h = Hk.shape[0] #len(canon)
    
    s = [0 for x in range(w)]
    Matrix = [[0 for x in range(h)] for y in range(w)] 
    Matrix = torch.tensor(Matrix)
    # print(Matrix.shape)
    for t in range((w)):
        s[t] = 0
        # 
        for j in range((h)):
            s[t] = s[t] + torch.exp(torch.matmul(Hq[t],torch.t(Hk[j])))
    for t in range(w):
        for n in range(h):
            Matrix[t][n] = torch.matmul(Hq[t], torch.t(Hk[n]))/s[t]
    # print(torch.tensor(Matrix).shape)
    return Matrix





