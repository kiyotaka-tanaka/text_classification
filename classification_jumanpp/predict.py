import tensorflow as tf
import numpy as np
import data_helper
from dict import Vocabulary
import argparse
from text_cnn_rnn import TextCNNRNN
import json
from pyknp import Jumanpp

parser = argparse.ArgumentParser()

parser.add_argument("--input_text",help="classify text",type=str,default="日本でのビジネス")
parser.add_argument("--path_to_model",help="model to use",type=str,default="./models/my-model.ckpt")

args = parser.parse_args()

jumanpp = Jumanpp()
classify_data = []

vocab = Vocabulary("data_use.txt")

result = jumanpp.analysis(args.input_text)
for mrph in result.mrph_list():
    word = mrph.midasi
    classify_data.append(vocab.stoi(word))

classify_data = data_helper.pad_one(classify_data,256,0)

with open("training_config.json") as f:
    params = json.load(f)

embedding_mat = np.load("./models/embedding.npy")
session_config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
sess = tf.Session(config= session_config)

with sess.as_default():
    cnn_rnn = TextCNNRNN(embedding_mat = embedding_mat,
                         non_static = params['non_static'],
                         hidden_unit = params['hidden_unit'],
                         sequence_length = 256,
                         max_pool_size = params['max_pool_size'],
                         num_classes = 2,
                         embedding_size = params['embedding_dim'],
                         filter_sizes = map(int,params['filter_sizes'].split(",")),
                         num_filters = params['num_filters'],
                         l2_reg_lambda = 0.0)

    def real_len(batches):
        return [np.ceil(np.argmin(batch + [0]) * 1.0/params['max_pool_size']) for batch in batches]

    saver = tf.train.Saver()
    saver.restore(sess,"./models/my-model.ckpt")

    prediction,scores = sess.run([cnn_rnn.predictions,cnn_rnn.scores],feed_dict={cnn_rnn.input_x:classify_data,cnn_rnn.dropout_keep_prob:1.0,cnn_rnn.batch_size:1,cnn_rnn.pad: np.zeros([1,1,params['embedding_dim'],1]),cnn_rnn.real_len: real_len(classify_data)})
    
print (prediction)
print (scores)
    
