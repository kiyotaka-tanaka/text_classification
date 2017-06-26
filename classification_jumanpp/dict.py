import sys
from pyknp import Jumanpp

jumanpp = Jumanpp()

class Vocabulary:
    def __init__(self,filename):
        self.fname = filename
        self.String2i={}
        self.i2String=[]
        if not self.fname is None:
            self.load_vocab()

    def stoi(self,word):
        if self.String2i.get(word):
            return self.String2i.get(word) - 1
        else:
            return self.String2i['<unk>']
    def itos(self,id):
        if id <len(self.i2String):
            return self.i2String[id]
        else:
            return '<unk>'

    def append_letter(self,word):
        if self.String2i.get(word):
            return
        else:
            self.i2String.append(word)
            id = len(self.i2String)
            self.String2i[word] = id
    def load_vocab(self):
        self.append_letter('None')
        self.append_letter('<unk>')
        with open(self.fname) as f:
            for line in f:
                line = str(line[:-3])
                result = jumanpp.analysis(line)
                for mprh in result.mrph_list():
                    self.append_letter(mprh.midasi)
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
                                                                                                                        
            
                    
