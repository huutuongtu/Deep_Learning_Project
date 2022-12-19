# from model import Net
from cProfile import label
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
# from help import beam_search_decoding
from model import Acoustic_Phonetic_Linguistic
from dataloader import MDD_Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
tokenizer = Wav2Vec2Processor.from_pretrained("pretrained_finetuned")

"""
"""
f = open("/home/tuht/train_wav2vec/loss.txt", 'a')
data = MDD_Dataset()
print(data)
net = Acoustic_Phonetic_Linguistic()
net.to('cuda')

net = torch.load('/home/tuht/train_wav2vec/MDD_Checkpoint/checkpoint_AdamW_16head_PL.pth')
# net = net.to('cpu')
train_loader = DataLoader(dataset=data,
                          batch_size=1,
                          shuffle=True,
                          num_workers=0)



# convert to an iterator and look at one random sample


ctc_loss = nn.CTCLoss(blank = 95)
# optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)
optimizer = optim.AdamW(net.parameters(), lr = 0.000001)
# optimizer = optim.SGD(net.parameters(), 0.01, momentum = 0.9)
# optimizer = torch.load('/home/tuht/train_wav2vec/MDD_Checkpoint/checkpoint_optim.pth')
for epoch in range(15):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader):
        acoustic, phonetic, linguistic, labels = data
        acoustic = acoustic.to('cuda')
        phonetic = phonetic.to('cuda')
        linguistic = linguistic.to('cuda')
        labels = labels.to('cuda')
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(acoustic, phonetic, linguistic)
        outputs = outputs.unsqueeze(1)
        input_lengths = outputs.shape
        target_lengths = labels.shape
        target = labels
        input_lengths = [input_lengths[0]]
        target_lengths =[target_lengths[1]]
        input_lengths = torch.tensor(input_lengths)
        target_lengths = torch.tensor(target_lengths)
        outputs = (F.log_softmax(outputs, dim=2))
        loss = ctc_loss(outputs, labels, input_lengths, target_lengths)
        print(loss)
        f.write("(" +str(epoch+5) + "," + str(i) + ")" + "  loss: " + str(loss) + "\n") 
        loss.backward()
        optimizer.step()
            
    torch.save(net, '/home/tuht/train_wav2vec/MDD_Checkpoint/checkpoint_AdamW_16head_PL.pth')
    # torch.save(optimizer, '/home/tuht/train_wav2vec/MDD_Checkpoint/checkpoint_optim_Adam.pth')
        
print('Finished Training')
