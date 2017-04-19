import tensorflow as tf
from dict import Vocabulary,letter_list,letter_list_text
#from model_rnn import RNN
from model import TextRnn
import numpy as np
import argparse
from data_helper import BatchLoader


parser = argparse.ArgumentParser()
parser.add_argument("--filename_vocab",help = "training_data",type = str,default="./data_new.txt")
parser.add_argument("--sequence_length",help = "letter_count",type =int,default=10)
parser.add_argument("--embedding_size",help= "size of embedding letter to vector",type = int, default=20)
parser.add_argument("--rnn_size",help="hidden layer size",type=int,default=21)
parser.add_argument("--model_path",help="model save path",type=str,default="./models/model.ckpt")
parser.add_argument("--max_epochs",help="max_epochs",type=int,default=300)
parser.add_argument("--vocab_size",help = "vocabulary",type=int)
parser.add_argument("--batch_size",help="batch_size",type =int,default=20)
parser.add_argument("--vocab_file",help="vocabulary file",type=str,default ="./data_new.txt")

args = parser.parse_args()

'''
def Solve_dictionary():
    vocab = Vocabulary()
'''

def onehot(x,size):
    ret = np.zeros((size),dtype= float)

    ret[x-1] = 1.0
    
    return ret

def make_batch():
    
    x_data

    pass

def train_here():

    ''''



    vocab = Vocabulary(args.filename_vocab)
    args.vocab_size = len(vocab.i2l)

    
    f = open(args.filename_vocab,"r")
    lines = f.readlines()
    f.close()
    
    vocab.save_vocab("dictionary.voc")
    '''

    
    #model = TextRnn(args.embedding_size,2,args.rnn_size,args.model_path,args.vocab_size,args.sequence_length,args.batch_size)


    model = TextRnn(args.embedding_size,args.vocab_file,5,args.rnn_size)

    #model.train("data_new.txt",200)

    #model = TextRnn(args.embedding_size,args.vocab_file,2,args.rnn_size)
    
    saver = tf.train.Saver()
    model.train("data.txt",200)

    
    saver.save(model.sess,'my-model.ckpt')
    
    #model.sess.run()

    
if __name__ == '__main__':   
    train_here()
