import tensorflow as tf 
import numpy as np 
from dict import Vocabulary

from dict import letter_list,letter_list_text



class TextRnn():
    

    def __init__(self,embedding_size,vocab_file,n_classes,rnn_size):
        
        self.real_len = tf.placeholder(tf.float32)
        
        self.input_data = tf.placeholder(tf.int32,[None,20])
        self.label_data = tf.placeholder(tf.int32,[None,n_classes])
        

        self.batch_size = tf.placeholder(tf.int32)


        self.vocab = Vocabulary(vocab_file)

        #vocab_size = len(self.vocab)
        vocab_size = len(self.vocab.i2l)

        self.embedding = tf.Variable(tf.truncated_normal([vocab_size,embedding_size]))
        




        self.w = tf.Variable(tf.truncated_normal([rnn_size,n_classes],stddev=0.1))
        self.b = tf.Variable(tf.truncated_normal([n_classes],stddev= 0.1))

        #######  state size has problem   #######

        

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
        self._initial_state = lstm_cell.zero_state(embedding_size, tf.float32)
        self.state = self._initial_state

        self.embedded_chars = tf.nn.embedding_lookup(self.embedding,self.input_data)
        
        print self.embedded_chars.get_shape
        
        #inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, reduced, pooled_concat)]
        inputs = [tf.squeeze(input_,[1])for input_ in tf.split(1,embedding_size,self.embedded_chars)]
        #lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size)
        self.output,self.state = tf.nn.rnn(lstm_cell,inputs,initial_state= self.state,sequence_length = self.real_len)
        
        self.logits = tf.matmul(self.output[-1],self.w) + self.b
        
        self.probs = tf.nn.softmax(self.logits)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.probs, self.label_data))

        
        self.opitimizer  = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(self.loss)
        
        self.sess = tf.InteractiveSession()


        

        #pass

    def run_line(self,x_list,y):
        
        self.state = self._initial_state
        
        
        for x in xlist:
            feed_dict_x = {self.input_data:x}
            
            self.sess.run(self.output,self.state,feed_dict = feed_dict_x)
        
            
        feed_dict_y = {self.label_data :y}
        self.sess.run(self.optimizer,feed_dict= feed_dict_y)

    def train(self,vocab_file,epoch_size):

        
        self.sess.run(tf.global_variables_initializer())


        vocab = Vocabulary(vocab_file)
        vocab.load_vocab()
        f = open(vocab_file,"rb")
        
        lines = f.readlines()

        for e in range(epoch_size):
            for line in lines:
                line = str(line)
                
                xlist ,y = letter_list_text(line)
                
                y = one_hot(y)
                x_list_use =[]
                for x in xlist:
                    x_list_use.append(vocab.l2i[x])

                
                sef.run_line(xlist_use,y)
                
                
                

            
                


                
