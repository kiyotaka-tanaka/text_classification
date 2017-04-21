import tensorflow as tf 
import numpy as np 
from dict import Vocabulary

from dict import letter_list,letter_list_text

'''

def onehot(x):
    if x == 0:
        return np.array([1.0,0.0])
    else:
        return np.array([0.0,1.0])


'''

class TextRnn():
    

    def __init__(self,embedding_size,vocab_file,n_classes,rnn_size):
        
        self.real_len = tf.placeholder(tf.float32)
        
        self.input_data = tf.placeholder(tf.int32,[None,1])
        self.label_data = tf.placeholder(tf.int32,[None,n_classes])
        self.embedding_size = embedding_size
        self.n_classes = n_classes
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
        
        inputs = [tf.squeeze(self.embedded_chars,[1]) ]

        #self.outputs,self.state = tf.nn.rnn(lstm_cell,inputs,initial_state= self.state,sequence_length = self.real_len)
        self.outputs,self.state = tf.nn.rnn(lstm_cell,inputs,initial_state= self._initial_state,sequence_length = self.real_len)

        
        output = self.outputs[-1]


        self.logits = tf.matmul(output,self.w) + self.b

        logits = self.logits[-1]
        self.probs = tf.nn.softmax(logits)
        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(self.probs, self.label_data))
        

        self.optimizer  = tf.train.AdamOptimizer(learning_rate = 0.00001).minimize(self.loss)
        
        self.sess = tf.InteractiveSession()



    def onehot(self,x,size):

        '''
        if x == 0:
            return np.array([1.0,0.0])
        else:
            return np.array([0.0,1.0])
        '''
        
        ret = np.zeros((size),dtype= float)
        ret[x-1] = 1.0
        return ret


    def run_line(self,x_list,y):
        
        self.state = self._initial_state

        while len(x_list) < self.embedding_size:
            x_list.append(0)
        x_list = np.array(x_list)


        x_list = x_list[0:self.embedding_size]
        x_list = np.resize(x_list,(len(x_list),1))


        feed_dict = {self.input_data:x_list,self.label_data:y,self.real_len:len(x_list)}
            
        
        
        
        _,loss= self.sess.run([self.optimizer,self.loss],feed_dict= feed_dict)
        return loss

    def train(self,vocab_file,epoch_size):

        self.sess.run(tf.initialize_all_variables())



        vocab = Vocabulary(vocab_file)
        vocab.load_vocab()
        f = open(vocab_file,"rb")
        
        lines = f.readlines()

        for e in range(epoch_size):
            print "epoch number is  "
            print e
            loss = 0.0
            for line in lines:
                
                line = str(line)
                
                if len(line) < 3:
                    continue
                xlist ,y = letter_list(line)

                
                '''
                if y == 0:
                    y = np.array([1.0,0.0])
                else:
                    y = np.array([0.0,1.0])
                '''
                size = self.n_classes
                
                
                y = self.onehot(y,size)

                x_list_use =[]
                for x in xlist:
                    x_list_use.append(vocab.l2i[x])
                y = np.resize(y,(1,len(y)))
                
                loss += self.run_line(x_list_use,y)
            print loss
                
                

            
                


                
