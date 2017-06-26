import MeCab 
import sys
import argparse
mecab = MeCab.Tagger("Choi")

class Vocabulary:
    def __init__(self,filename):
        self.fname = filename
        self.String2i = {}
        self.i2String = []
        if not self.fname is None:
            self.load_vocab()
    def stoi(self, word):
        if word in self.String2i:
            return self.String2i[word]
        return self.String2i['<unk>']
    def itos(self, id):
        if id < len(self.i2String):
            return self.i2String[id]
        return '<unk>'
    def append_letter(self,word):
        if word in self.String2i:
            return 
        self.i2String.append(word)
        id = len(self.String2i) 
        self.String2i[word] = id
    def load_vocab(self):
        self.append_letter('None')
        self.append_letter('<unk>')
        with open(self.fname,"r") as f:
            for line in f:
                line = line[:-3]
                line = str(line)
                node = mecab.parseToNode(line)
                while node:
                    self.append_letter(node.surface)
                    node = node.next
    def save_Vocab(self,vocabname):
        f = open(vocabname,"w")
        for line in self.i2String:
            f.write(line + "\n")
    #### static load method #######
def load_from_file(filename):
    vocab = Vocabulary(None)
    with open(filename,"r") as f:
        for line in f:
            vocab.append_letter(line)
        return vocab
        
            
