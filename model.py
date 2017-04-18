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
        

        print "embedding size"
        print self.embedded_chars.get_shape()
        
        #inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, reduced, pooled_concat)]
        #inputs = [tf.squeeze(input_,[1])for input_ in tf.split(embedding_size,self.embedded_chars)]
        inputs = [tf.squeeze(self.embedded_chars,[1]) ]
        print "input size input size"
        print len(inputs) ,inputs[0].get_shape()
        #lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size)
        self.outputs,self.state = tf.nn.rnn(lstm_cell,inputs,initial_state= self.state,sequence_length = self.real_len)

        
        print "bias shape bias shape bias shape"
        
        print self.b.get_shape()
        
        output = self.outputs[-1]

        print "output size "
        print output.get_shape()
        self.logits = tf.matmul(output,self.w) + self.b
        print "logits shape logits shape"
        print self.logits.get_shape()
        logits = self.logits[-1]
        self.probs = tf.nn.softmax(logits)
        #self.score = tf.argmax(self.probs,1)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.probs, self.label_data))
        
        print "last last last last last last last "
        print self.logits.get_shape(),self.label_data.get_shape
        self.optimizer  = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(self.loss)
        
        self.sess = tf.InteractiveSession()


        

        #pass


    def onehot(x):
        if x == 0:
            return np.array([1.0,0.0])
        else:
            return np.array([0.0,1.0])

    def run_line(self,x_list,y):
        
        self.state = self._initial_state
        print "from run line"
        print x_list
        print len(x_list)
        while len(x_list) < 20:
            x_list.append(0)
        x_list = np.array(x_list)


        x_list = x_list[0:20]
        x_list = np.resize(x_list,(len(x_list),1))
        print x_list.shape
        
        feed_dict = {self.input_data:x_list,self.label_data:y,self.real_len:len(x_list)}
            
        #self.sess.run(self.output,self.state,feed_dict = feed_dict_x)
        
        
        
        self.sess.run(self.optimizer,feed_dict= feed_dict)

    def train(self,vocab_file,epoch_size):

        self.sess.run(tf.initialize_all_variables())
        #self.sess.run(tf.global_variables_initializer())


        vocab = Vocabulary(vocab_file)
        vocab.load_vocab()
        f = open(vocab_file,"rb")
        
        lines = f.readlines()

        for e in range(epoch_size):
            for line in lines:
                print line

                print "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
                line = str(line)
                
                xlist ,y = letter_list(line)
                
                print y
                print type(y)
                
                
                if y == 0:
                    y = np.array([1.0,0.0])
                else:
                    y = np.array([0.0,1.0])
            
                
                x_list_use =[]
                for x in xlist:
                    x_list_use.append(vocab.l2i[x])
                y = np.resize(y,(1,2))
                
                self.run_line(x_list_use,y)
                
                
                

            
                


                
