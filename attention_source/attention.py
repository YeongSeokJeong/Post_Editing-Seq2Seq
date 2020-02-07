import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from konlpy.tag import Kkma
import time
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
from addition_data import * 
import pickle
BATCH_SIZE = 128
embedding_dim = 300
units =  500
# 인코더 디코더의 순환신경망에서 사용할 은닉층 차원수 

data = pd.read_csv("data_new7.csv",encoding = 'cp949')

input_data = data.iloc[:,0].to_list()
output_data = data.iloc[:,1].to_list()

input_vocab,output_vocab = set(),set()
input_max_len = 0
output_max_len = 0
kkma = Kkma()

morphs_train_input = []
morphs_train_output = []

morphs_val_input = []
morphs_val_output = []

morphs_test_input = []
morphs_test_output = []

with open("trian_input_data.pickle", "rb") as fr:
    morphs_train_input = pickle.load(fr)

with open("train_output_data.pickle", 'rb') as fr:
    morphs_train_output = pickle.load(fr)

with open('val_input_tokens.pickle', 'rb') as fr:
    morphs_val_input = pickle.load(fr)

with open('val_output_tokens.pickle', 'rb') as fr:
    morphs_val_output = pickle.load(fr) 

with open('test_input_tokens.pickle', 'rb') as fr:
    morphs_test_input = pickle.load(fr)

with open('test_output_tokens.pickle', 'rb') as fr:
    morphs_test_output = pickle.load(fr)

with open('input_vocab.pickle', 'rb') as fr:
    input_vocab = pickle.load(fr)

with open('output_vocab.pickle', 'rb') as fr:
    output_vocab = pickle.load(fr)



dic_input_vocab = {word:i for i, word in enumerate(input_vocab)}
dic_output_vocab = {word:i for i, word in enumerate(output_vocab)}

train_input_tokens = []
train_output_tokens = []

val_input_tokens = []
val_output_tokens = []

test_input_tokens = []
test_output_tokens = []

for i, (t_input, t_output) in enumerate(zip(morphs_train_input, morphs_train_output)):
    input_steplen = len(t_input)
    output_steplen = len(t_output)

    step_input = [dic_input_vocab["<start>"]] + [dic_input_vocab[word] for word in t_input] + [dic_input_vocab["<end>"]]
    step_output =  [dic_output_vocab["<start>"] + [dic_output_vocab[word] for word in t_output] + [dic_output_vocab["<end>"]]]

    train_input_tokens.append(step_input)
    train_output_tokens.append(step_output)
 
    input_max_len = input_max_len if input_max_len > input_steplen else input_steplen
    output_max_len = output_max_len if output_max_len > output_steplen else output_steplen

for i, (v_input, v_output) in enumerate(zip(morphs_val_input, morphs_val_output)):
    input_steplen = len(t_input)
    output_steplen = len(t_output)

    step_input = [dic_input_vocab["<start>"]] + [dic_input_vocab[word] for word in v_input] + [dic_input_vocab["<end>"]]
    step_output =  [dic_output_vocab["<start>"] + [dic_output_vocab[word] for word in v_output] + [dic_output_vocab["<end>"]]]
    
    val_input_tokens.append(step_input)
    val_output_tokens.append(step_output)

    input_max_len = input_max_len if input_max_len > input_steplen else input_steplen
    output_max_len = output_max_len if output_max_len > output_steplen else output_steplen

for i, (t_input, t_output) in enumerate(zip(morphs_test_input, morphs_test_output)):
    input_steplen = len(t_input)
    output_steplen = len(t_output)

    step_input = [dic_input_vocab["<start>"]] + [dic_input_vocab[word] for word in train_input] + [dic_input_vocab["<end>"]]
    step_output =  [dic_output_vocab["<start>"] + [dic_output_vocab[word] for word in t_output] + [dic_output_vocab["<end>"]]]
    
    test_input_tokens.append(step_input)
    test_output_tokens.append(step_output)

    input_max_len = input_max_len if input_max_len > input_steplen else input_steplen
    output_max_len = output_max_len if output_max_len > output_steplen else output_steplen

train_input_tokens = pad_sequences(train_input_tokens, input_max_len, padding = 'post')
train_output_tokens = pad_sequences(train_output_tokens, output_max_len, padding = 'post')

val_input_tokens = pad_sequences(val_input_tokens, input_max_len, padding = 'post')
val_output_tokens = pad_sequences(val_output_tokens, output_max_len, padding = 'post')

test_input_tokens = pad_sequences(test_input_tokens, input_max_len, padding = 'post')
test_output_tokens = pad_sequences(test_output_tokens, output_max_len, padding = 'post')
# keras의 pad_sequences 함수를 사용하여 문장을 패딩한다. 이때 패딩된 값은 0 즉 '<p>'로 패딩된다.


class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state
  # Encoder 선언후 호출시에 실행되는 함수

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))
  # Encoder의 순환 신경망의 hidden_state를 초기화 하기위한 함수 정의
encoder = Encoder(len(input_vocab), embedding_dim = 300, enc_units = 200, batch_sz = BATCH_SIZE)

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
        # self.W1(values) => (batch_size, seq_len, units)
        # self.W2(hidden_with_time_axis) => (batch_size, 1, units)

        # score shape => (batch_size, seq_len, 1)
        attention_weights = tf.nn.softmax(score, axis = 1)
        # softmax encoder의 단어의 중요도를 각각 얻기 위해 사용한다. 
        context_vector = attention_weights*values
        context_vector = tf.reduce_sum(context_vector, axis = 1)
        
        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self,vocab_size, embedding_dim, dec_units, batch_sz):
        # Model은 인풋과 아웃풋 텐서를 음 
        super(Decoder,self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                      return_sequences=True,
                                      return_state = True,
                                      recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)
        
    def call(self,x,hidden, enc_output):
        context_vector,attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector,1),x], axis = -1)
        # context_vector 와 단순 차원 합을 한다. (이어 붙임)
        output,state = self.gru(x)
        output = tf.reshape(output,(-1,output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights

optimizer = tf.keras.optimizers.Adam()
# 최적화 함수 정의
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction = 'none')
def loss_function(real,pred):
    mask = tf.math.logical_not(tf.math.equal(real,0))
    loss_ = loss_object(real,pred)
    
    mask = tf.cast(mask,dtype = loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)
# 손실함수 정의

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
# 모델 훈련을 위한 함수 정의

def validation_loss(val_input = val_input,  val_output = val_output):
    total_loss = 0

    for batch in range( int(len(val_input)/BATCH_SIZE)):
        loss = 0

        test_input = val_input[BATCH_SIZE * batch: BATCH_SIZE  * (batch + 1)]
        test_output = val_output[BATCH_SIZE * batch: BATCH_SIZE  * (batch + 1)]
        
        inputs = tf.convert_to_tensor(test_input)
        hidden = encoder.initialize_hidden_state()

        enc_out, enc_hidden = encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([dic_input_vocab['<start>']] * BATCH_SIZE, 1)

        for t in range(inputs.shape[1]):
            predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

            attention_weights = tf.reshape(attention_weights, (-1,))

            loss += loss_function(test_output[:, t], predictions) 

            predictions = tf.argmax(predictions, 1)
            dec_input = tf.expand_dims(predictions, 1)
        loss = loss/inputs.shape[1]
                
        total_loss += loss
    total_loss /= int(len(val_input)/BATCH_SIZE)
    return total_loss


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

        if batch % 10 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))
    # saving (checkpoint) the model every 2 epochs
    
    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    print('Validation Loss {:.4f}\n'. format(validation_loss()))


def evaluate(sentence):
    inp = preprocess(sentence)

    inputs = tf.convert_to_tensor(inp)

    result = ''
    
    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([dic_input_vocab['<start>']], 0)

    for t in range(output_max_len):
        predictions, dec_hidden, attention_weights = decoder(dec_input, 
                                                             dec_hidden,
                                                             enc_out)
        attention_weights = tf.reshape(attention_weights, (-1,))
        

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += output_vocab[predicted_id] + ' '
        
        if output_vocab[predicted_id] == '<end>':
            return result, sentence
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence

def return_hangul(sentence):
    sen = []
    for word in sentence:
        sen.append(input_vocab[input_vocab.index[word]])
    return sen
def translate(sentence):
    sentence = return_hangul(sentence)
    result, sentence = evaluate(sentence)
    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))

for i in range(len(test_input_tokens)//100):
    print(test_input_tokens[i*100])