import os
import sys
import json
import pickle
import numpy as np
import gensim as gs
import pandas as pd
from dict import Vocabulary
from pyknp import Jumanpp

jumanpp = Jumanpp()

def load_embeddings(vocabulary):
    word_embeddings ={}
    for word in vocabulary:
        word_embeddings[word] = np.random.uniform(-0.25,0.25,300)
    return word_embeddings
def pad_data(data,size,pad_index):
    new_data = []
    for data_ in data:
        if len(data_) >= size:
            data_ = data_[:size]
        else:
            while len(data_) < size:
                data_.append(pad_index)
        new_data.append(data_)

    return new_data

def batch_iter(data,batch_size,num_epochs, shuffle=True):
    data= np.array(data)
    data_size = len(data)
    num_batches_per_epochs = int(data_size/batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epochs):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1)*batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def load_data(filename):
    vocab = Vocabulary(filename)
    with open(filename) as f:
        lines = f.readlines()
    f.close()
    data = []
    label = []
    for line in lines:
        #print (line)
        line = line.strip()
        nline = str(line[:-3])
        #print (nline.decode('utf-8','replace'))
        label.append(int(line[-1]))
        data1 =[]
        result = jumanpp.analysis(nline)
        for mprh in result.mrph_list():
            data1.append(vocab.stoi(mprh.midasi))
        data.append(data1)

    return data,label

def one_hot(label,label_size):
    arr = np.zeros([len(label),label_size],dtype = np.float32)
    for number in range(len(label)):
        arr[number][label[number]-1] = 1.0
    return arr

def pad_one(data,size,pad_index):
    
    if len(data) >= size:
        data = data[:size]
    else:
        while len(data) < size:
            data.append(pad_index)

    arr = np.array(data)
    arr = np.resize(arr,(1,size))
    return arr


if __name__=='__main__':
    data,label = load_data("data_use.txt")
