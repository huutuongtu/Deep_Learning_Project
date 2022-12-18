import torch
from torch import nn
import json
import json
import pandas as pd
import re
import csv
import numpy as np
# writer = csv.writer(f)

def clean_corpus(str1):
    res1 = ""
    for i in str1:
        if i.isalpha() or i==" ":
            res1 = "".join([res1, i])
    return res1
d = 0
with open('/home/tuht/train_wav2vec/vocab.json') as f:
    d = json.load(f)
def text_to_tensor(str):
    text = str
    text = text.lower()
    text = clean_corpus(text)
    text_list = []
    for idex in text:
        if idex!=" ":
            text_list.append(d[idex])
        elif idex==" ":
            text_list.append(d["|"])
    return text_list

key_list = list(d.keys())
val_list = list(d.values())

def tensor_to_text(ts):
    int_to_text = ts
    res = []
    for i in int_to_text:
        position = val_list.index(i)
        res.append(key_list[position])
    return res

def test_tensor_to_text(ts):
    int_to_text = ts
    res = []
    for i in int_to_text:
        position = val_list.index(i)
        print(key_list[position], end = '')
    

# txt = [29, 29, 91, 56, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 91, 29, 29, 56, 29, 91, 91, 22, 29, 29, 29, 91, 91, 91, 29, 78, 38, 29, 29, 29, 91, 29, 29, 29, 29, 29, 29, 29, 38, 29, 91, 29, 29, 29, 91, 29, 91, 29, 56, 91, 29, 29, 29, 29, 29, 29, 91, 29, 29, 29, 29, 29, 29, 29, 56, 29, 91, 29, 29, 91, 29, 29, 29, 91, 91, 29, 56, 29, 29, 29, 29, 29, 91, 29, 29, 29, 29, 29, 29, 29, 91, 29, 91, 29, 29, 29, 29, 29, 29, 91, 29, 29, 91, 91, 29, 29, 29, 91, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 91, 29, 91, 29, 29, 29, 91, 29, 91, 29, 29, 29, 29, 29, 29, 29, 29, 56, 91, 29, 29, 29, 29, 29, 78, 29, 91, 91, 29, 29, 29, 29, 91, 91, 91, 91, 91, 12, 29, 29, 29, 29, 91, 29, 29, 38, 29, 91, 29, 91, 29, 29, 91, 29, 29, 29, 91]
# print(test_tensor_to_text(txt))
