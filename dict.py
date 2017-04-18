# -*- coding: utf-8 -*-

import argparse
import sys, time


#from model import LetterClassifyer
'''
def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode')
    parser.add_argument('file')
    parser.add_argument('--embed', default=200, type=int)
    parser.add_argument('--vocab', default=3000, type=int)
    parser.add_argument('--hidden', default=1000, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--model', default="model")
    parser.add_argument('--classes', default=2)
    parser.add_argument('--use-gpu', action='store_true', default=False)
    parser.add_argument('--unchain', action='store_true', default=False)
    args = parser.parse_args()
    return args

'''

# ファイルから1文字単位の列とラベルを取得
def letter_list(string):
    #with open(fname) as f:
    #for l in string:
    l = list(string)
    body = l[:-3]
    val = int(l[-2])
    #x = (''.join(body.split()))
    body.append('</s>')
    #yield x, val
    return body,val
def letter_list_text(t):
    x = list(''.join(t.split()))
    x.append('</s>')
    return x
# 
class Vocabulary:
    def __init__(self, fname):
        self.fname = fname
        self.l2i = {}
        self.i2l = []
        if not fname is None:
            self.load_vocab()
    def stoi(self, letter):
        if letter in self.l2i:
            return self.l2i[letter]
        return self.l2i['<unk>']
    def itos(self, id):
        if id < len(self.i2l):
            return self.i2l[id]
        return '<unk>'
            
    def append_letter(self, l):
        if l in self.l2i:
            return
        self.i2l.append(l)
        id = len(self.i2l) -1
        self.l2i[l] = id
    def load_vocab(self):
        self.append_letter('<unk>')
        self.append_letter('<s>')
        self.append_letter('</s>')
        with open(self.fname) as f:
            for line in f:
                nline = line[:-3]
                for l in nline:
                    self.append_letter(l)

    def save_vocab(self, filename):
        with open(filename, 'w') as f:
            for l in self.i2l:
                f.write(l + "\n")
    @staticmethod
    def load_from_file(filename):
        vocab = Vocabulary(None)
        with open(filename) as f:
            for l in f:
                l = l[:-1]
                vocab.append_letter(l)
        return vocab

