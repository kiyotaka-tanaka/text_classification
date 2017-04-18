from dict import Vocabulary,letter_list

import numpy as np




class BatchLoader():
    def __init__(self,batch_size,seq_length,vocab_file):
        self.batch_size = batch_size
        self.seq_length = seq_length
        #pass

        self.vocab = Vocabulary(vocab_file)
    
        self.length = len(self.vocab) +1

    def create_batches(self,filename):
        #pass
        
        x_data ,y_label = self.load_data(filename)

        batch_count =  len(x_data)/self.bach_size + 1

        
        
    def next_batch(self):
        pass
        
    def reset_batch_pointer(self):
        pass


    def Conver2int():
        pass

    def load_data(self,filename):
        vocab = Vocabulary(filename)
        f = open(filename,"r")
    
        lines = f.readlines()
        print ("data_length is %d") %(len(lines))
        x_data=[]
        y_label=[]
        for line in lines:
            letters,label = letter_list(line)
            #print label
            data_line =[]
        for letter in letters:
            data_line.append(vocab.stoi(letter))
            x_data.append(data_line)
            y_label.append(label)

        return x_data,y_label



if __name__ == '__main__':
    load_data("data.txt")
