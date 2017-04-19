import tensorflow as tf
import argparse
import numpy as np

from model import TextRnn
from dict import Vocabulary

from dict import letter_list,letter_list_text

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

model = TextRnn(args.embedding_size,args.vocab_file,5,args.rnn_size)

sess = tf.InteractiveSession()
saver = tf.train.Saver()

saver.restore(sess,"./models/my-model.ckpt")


