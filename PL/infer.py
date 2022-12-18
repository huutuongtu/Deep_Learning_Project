# import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import time
import os
import librosa
import torch.nn as nn

os.environ['CUDA_VISBLE_DEVICES'] = '0'
tokenizer = Wav2Vec2Processor.from_pretrained("pretrained_finetuned")
model = Wav2Vec2ForCTC.from_pretrained("pretrained_finetuned").eval()
newmodel = torch.nn.Sequential(*(list(model.children())[:-2]))
newmodel.eval().to('cuda')
# print(model)
# model_extract_feature = torch.nn.Sequential((list(model.children())[0]))
# print(type(model_extract_feature))
# print(model.lm_head.weight.shape)

def phonetic_embedding(wav_dir):
    # a = time.perf_counter()
    link = wav_dir
    y, sr = librosa.load(link, sr=16000)
    y_16k = librosa.resample(y, sr, 16000)
    audio_input = librosa.to_mono(y_16k)
    input_values = tokenizer(audio_input, return_tensors="pt", sampling_rate=16000).input_values.to('cuda')
    # print(model(input_values).logits.shape)
    # a = time.perf_counter() - a
    return ((newmodel(input_values).last_hidden_state))
    # return newmodel(input_values).last_hidden_state, a
