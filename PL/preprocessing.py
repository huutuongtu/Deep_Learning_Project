import numpy as np
from infer import phonetic_embedding
import pandas as pd

data = pd.read_csv("/home/tuht/train_wav2vec/vi/vi/validated.csv", sep='\t')
for i in range(len(data)):
    phonetic = phonetic_embedding(data['path'][i])
    phonetic = phonetic.squeeze(0)
    x = phonetic.detach().cpu().numpy()
    p = data['path'][i].split(".")[0]
    p = p.split("/")[7]
    # print(p)
    print(i)
    np.save('/home/tuht/train_wav2vec/phonetic/' + str(p) + ".npy", x)