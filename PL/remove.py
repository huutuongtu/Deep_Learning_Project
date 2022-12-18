import glob
import pandas as pd
import librosa
data = pd.read_csv('/home/lab/Data_SV/train_wav2vec/vi/vi/validated.csv',sep='\t')
for i in range(len(data)):
    if librosa.get_duration(filename = data['path'][i])<0.1:
        print(data['path'][i])