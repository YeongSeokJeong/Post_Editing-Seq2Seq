import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from konlpy.tag import Kkma
import time
from tensorflow.keras.layers import *
from tensorflow.keras import Model

BATCH_SIZE = 128
embedding_dim = 300
units = 128

data = pd.read_csv("data_new7.csv",encoding = 'cp949')

input_data = data.iloc[:,0].to_list()[:10000]
output_data = data.iloc[:,1].to_list()[:10000]

input_vocab,output_vocab = set(),set()
input_max_len = 0
output_max_len = 0
kkma = Kkma()
morphs_input_sentence = []
morphs_output_sentence = []

for input_sentence,output_sentence in zip(input_data,output_data):
    input_sentence = kkma.morphs(input_sentence)
    output_sentence = kkma.morphs(output_sentence)
    
    morphs_input_sentence.append(input_sentence)
    morphs_output_sentence.append(output_sentence)
    
    input_vocab.update(input_sentence)
    output_vocab.update(output_sentence)
    
    input_steplen = len(input_sentence)
    output_steplen = len(output_sentence)
    
    input_max_len = input_max_len if input_max_len > input_steplen else input_steplen
    output_max_len = output_max_len if output_max_len > output_steplen else output_steplen

input_vocab = ['<start>', '<end>'] + list(input_vocab)
output_vocab = ['<start>', '<end>'] + list(output_vocab)

input_vocab_size = len(input_vocab)
output_vocab_size = len(output_vocab)

dic_input_vocab = {word:index for index,word in enumerate(input_vocab)}
dic_output_vocab = {word:index for index,word in enumerate(output_vocab)}

input_tokens = []
output_tokens = []
for i,(inp_sen, targ_sen) in enumerate(zip(morphs_input_sentence, morphs_output_sentence)):
    s_input_token = [[dic_input_vocab['<start>']] + [dic_input_vocab[word] for word in inp_sen] + [dic_output_vocab['<end>']]]
    s_output_token = [[dic_output_vocab['<start>']] + [dic_output_vocab[word] for word in targ_sen] + [dic_output_vocab["<end>"]]]
    
    input_tokens.append(s_input_token)
    output_tokens.append(s_output_token)

input_tokens = [[dic_input_vocab[word] for word in sentence] for sentence in morphs_input_sentence]
output_tokens = [[dic_output_vocab[word] for word in sentence] for sentence in morphs_output_sentence]


input_tokens = pad_sequences(input_tokens, input_max_len)
output_tokens = pad_sequences(output_tokens, output_max_len)

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, gru_hidden, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.gru_hidden = gru_hidden
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.gru_hidden,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.gru_hidden))


encoder = Encoder(len(input_vocab), embedding_dim = embedding_dim, gru_hidden = units, batch_sz = BATCH_SIZE)

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self,units):
        super(BahdanauAttention,self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, query,values):
        #query => encoder hidden
        #values => decoder hidden
        hidden_with_time_axis = tf.expand_dims(query, 1)
        # hidden_with_time_axis의 shape은 (batch_size, 1, hidden_size)이다.
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        # score shape => (batch_size, seq_len, 1)
        attention_weights = tf.nn.softmax(score, axis = 1)
        context_vector = attention_weights*values
        context_vector = tf.reduce_sum(context_vector, axis = 1)
        
        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self,vocab_size, embedding_dim, gru_hidden, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.gru_hidden = gru_hidden
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.gru_hidden,
                                      return_sequences=True,
                                      return_state = True,
                                      recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.gru_hidden)
        
    def call(self, x, gru_hidden, enc_output):
        context_vector,attention_weights = self.attention(gru_hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector,1),x], axis = -1)
        output,state = self.gru(x)
        output = tf.reshape(output,(-1,output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights

decoder = Decoder(len(output_vocab), embedding_dim, units, BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction = 'none')
def loss_function(real,pred):
    mask = tf.math.logical_not(tf.math.equal(real,0))
    loss_ = loss_object(real,pred)
    
    mask = tf.cast(mask,dtype = loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([dic_output_vocab['<start>']] * BATCH_SIZE, 1)

        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)

            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

steps_per_epoch = len(input_tokens)//BATCH_SIZE

EPOCHS = 10

for epoch in range(EPOCHS):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for batch in range(len(input_tokens)//BATCH_SIZE):
        batch_input = input_tokens[batch*BATCH_SIZE: (batch+1)*BATCH_SIZE]
        batch_output = output_tokens[batch*BATCH_SIZE: (batch+1)*BATCH_SIZE]

        batch_loss = train_step(batch_input, batch_output, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))
    # saving (checkpoint) the model every 2 epochs
    
    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))