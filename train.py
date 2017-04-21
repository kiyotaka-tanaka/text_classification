import tensorflow as tf
from dict import Vocabulary,letter_list,letter_list_text
#from model_rnn import RNN
from model import TextRnn
import numpy as np
import argparse
from data_helper import BatchLoader


parser = argparse.ArgumentParser()
parser.add_argument("--filename_vocab",help = "training_data",type = str,default="./data_new.txt")
#parser.add_argument("--sequence_length",help = "letter_count",type =int,default=10)


parser.add_argument("--embedding_size",help= "size of embedding letter to vector and seq length",type = int, default=20)
parser.add_argument("--rnn_size",help="hidden layer size",type=int,default=20)
parser.add_argument("--model_path",help="model save path",type=str,default="./models/model.ckpt")
parser.add_argument("--max_epochs",help="max_epochs",type=int,default=300)
#"parser.add_argument("--vocab_size",help = "vocabulary",type=int)
#parser.add_argument("--batch_size",help="batch_size",type =int,default=20)
parser.add_argument("--vocab_file",help="vocabulary file",type=str,default ="./data_use.txt")
parser.add_argument("--n_classes",help="label size ",type=int,default=5)
parser.add_argument("--epoch_size",help ="epoch size ", type= int, default = 100)

args = parser.parse_args()





def train_here():

    ''''



    vocab = Vocabulary(args.filename_vocab)
    args.vocab_size = len(vocab.i2l)

    
    f = open(args.filename_vocab,"r")
    lines = f.readlines()
    f.close()
    
    vocab.save_vocab("dictionary.voc")
    '''

    



    model = TextRnn(args.embedding_size,args.vocab_file,args.n_classes,args.rnn_size)
    



    
    saver = tf.train.Saver()
    model.train(args.vocab_file,args.epoch_size)

    
    saver.save(model.sess,'./models/my-model.ckpt')
    


    
if __name__ == '__main__':   
    train_here()
