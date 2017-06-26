import tensorflow as tf
import numpy as np
from text_cnn_rnn import TextCNNRNN
from dict import Vocabulary
import data_helper
import argparse
import json
import time
import os
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()

parser.add_argument("--embedding_size",type = int,help="embedding size",default=300)
parser.add_argument("--hidden_unit",type= int,help="hidden unit size",default=256)
parser.add_argument("--sequence_length",type=int,help="length of sentence",default=256)
parser.add_argument("--train_data",type=str,help="trianing data", default="data_use.txt")
parser.add_argument("--config_file",type=str,help = "training config",default="training_config.json")

args = parser.parse_args()

vocabulary = Vocabulary(args.train_data)
word_embedding = data_helper.load_embeddings(vocabulary.String2i)
embedding_mat = [word_embedding[word] for index,word in enumerate(vocabulary.i2String)]
embedding_mat = np.array(embedding_mat,np.float32)

'''
print embedding_mat.shape
print embedding_mat[20]
'''
with open(args.config_file) as f:
    params = json.load(f)


#print params

'''
{u'hidden_unit': 300, u'l2_reg_lambda': 0.0, u'dropout_keep_prob': 0.5, u'num_filters': 128, u'max_pool_size': 4, u'embedding_dim': 300, u'batch_size': 256, u'filter_sizes': u'3,4,5', u'evaluate_every': 100, u'non_static': False, u'num_epochs': 1}

'''

timestamp = str(int(time.time()))

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(allow_soft_placement = True, log_device_placement=False)
    sess = tf.Session(config = session_conf)
    with sess.as_default():
        cnn_rnn = TextCNNRNN(embedding_mat = embedding_mat,
                             non_static = params['non_static'],
                             hidden_unit = params['hidden_unit'],
                             sequence_length = 256,
                             max_pool_size = params['max_pool_size'],
                             num_classes = 2,
                             embedding_size = params['embedding_dim'],
                             filter_sizes =map(int,params['filter_sizes'].split(",")),
                             num_filters = params['num_filters'],
                             l2_reg_lambda = 0.0)
        
        global_step = tf.Variable(0,name='global_step',trainable= False)
        optimizer = tf.train.RMSPropOptimizer(1e-3,decay=0.9)
        grads_and_vars = optimizer.compute_gradients(cnn_rnn.loss)

        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        
        
    
	# Checkpoint files will be saved in this directory during training
	checkpoint_dir = './checkpoints_' + timestamp + '/'
	if os.path.exists(checkpoint_dir):
	    shutil.rmtree(checkpoint_dir)
	#os.makedirs(checkpoint_dir)
	#checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
    
	def real_len(batches):
	    return [np.ceil(np.argmin(batch + [0]) * 1.0 / params['max_pool_size']) for batch in batches]
    
	def train_step(x_batch, y_batch):
	    feed_dict = {
		cnn_rnn.input_x: x_batch,
	        cnn_rnn.input_y: y_batch,
		cnn_rnn.dropout_keep_prob: params['dropout_keep_prob'],
	        cnn_rnn.batch_size: len(x_batch),
		cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
					cnn_rnn.real_len: real_len(x_batch),
				}
	    _, step, loss, accuracy = sess.run([train_op, global_step, cnn_rnn.loss, cnn_rnn.accuracy], feed_dict)

            print loss
	def dev_step(x_batch, y_batch):
	    feed_dict = {
		cnn_rnn.input_x: x_batch,
		cnn_rnn.input_y: y_batch,
		cnn_rnn.dropout_keep_prob: 1.0,
		cnn_rnn.batch_size: len(x_batch),
		cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
		cnn_rnn.real_len: real_len(x_batch),
	    }
	    step, loss, accuracy, num_correct, predictions = sess.run(
		[global_step, cnn_rnn.loss, cnn_rnn.accuracy, cnn_rnn.num_correct, cnn_rnn.predictions], feed_dict)
	    return accuracy, loss, num_correct, predictions

        saver = tf.train.Saver(tf.all_variables())
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
                        

        data,label = data_helper.load_data("data_use.txt","")
        label =data_helper.onehot(label,2)
        data = data_helper.pad_data(data,256,0)

        x_train,x_dev,y_train,y_dev = train_test_split(data,label,test_size=0.1)

        train_batches = data_helper.batch_iter(list(zip(x_train,y_train)),100,params['num_epochs'],shuffle=True)
        best_accuracy  = 0.0
        for train_batch in train_batches:
            x_train,y_train  = zip(*train_batch)
            train_step(x_train,y_train)
            current_step = tf.train.global_step(sess,global_step)

            if current_step % params['evaluate_every'] == 0:
                dev_batches = data_helper.batch_iter(list(zip(x_dev,y_dev)),100,1)
                total_dev_correct = 0
                for dev_batch in dev_batches:
                    x_dev_batch,y_dev_batch = zip(*dev_batch)
                    acc,loss,num_dev_correct,predictions = dev_step(x_dev_batch,y_dev_batch)
                    total_dev_correct += num_dev_correct

                accuracy = float(total_dev_correct)/float(len(y_dev))
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    saver.save(sess,"./models/my-model.ckpt")

        
        np.save("./models/embedding.npy",embedding_mat)
        
