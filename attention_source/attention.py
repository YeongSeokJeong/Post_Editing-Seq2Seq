import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from konlpy.tag import Kkma
import time
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
BATCH_SIZE = 128
embedding_dim = 300
units =  500
# 인코더 디코더의 순환신경망에서 사용할 은닉층 차원수 

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
    # Kkma 분석기를 사용한 형태소 분석
    morphs_input_sentence.append(input_sentence)
    morphs_output_sentence.append(output_sentence)

    input_vocab.update(input_sentence)
    output_vocab.update(output_sentence)
    # 단어 사전을 만들기 위한 단어 업데이트

    input_steplen = len(input_sentence)
    output_steplen = len(output_sentence)
    # 현재 time step의 문장의 길이를 저장

    input_max_len = input_max_len if input_max_len > input_steplen else input_steplen
    output_max_len = output_max_len if output_max_len > output_steplen else output_steplen
    # 문장의 최장 길이를 저장하기 위한 변수

input_vocab = ['<p>', '<start>', '<end>'] + list(input_vocab)
output_vocab = ['<p>','<start>', '<end>'] + list(output_vocab)
# vocab에 패딩 단어, 시작 단어, 끝 단어를 추가한다.

dic_input_vocab = {word:index for index,word in enumerate(input_vocab)}
dic_output_vocab = {word:index for index,word in enumerate(output_vocab)}
# 이를 딕셔너리 형태로 만들어 저장한다. 

input_tokens = []
output_tokens = []
# 입력 문장, 출력 문장의 문장쌍을 저장할 각각의 변수 선언.

for i,(inp_sen, targ_sen) in enumerate(zip(morphs_input_sentence, morphs_output_sentence)):
    s_input_token = [[dic_input_vocab['<start>']] + [dic_input_vocab[word] for word in inp_sen] + [dic_output_vocab['<end>']]]
    s_output_token = [[dic_output_vocab['<start>']] + [dic_output_vocab[word] for word in targ_sen] + [dic_output_vocab["<end>"]]]
    # 각 문장의 시작과 끝을 추가하기 위한 <'start'>, <'end'>를 추가한다. 

    input_tokens.append(s_input_token)
    output_tokens.append(s_output_token)
    # 추가된 각 문장을 변수에 추가한다.

input_tokens = pad_sequences(input_tokens, input_max_len, padding = 'post')
output_tokens = pad_sequences(output_tokens, output_max_len, padding = 'post')
# keras의 pad_sequences 함수를 사용하여 문장을 패딩한다. 이때 패딩된 값은 0 즉 '<p>'로 패딩된다.

train_input, train_output, val_input, val_output = train_test_split(input_tokens,
                                                                    output_tokens,
                                                                    test_size = 0.1,
                                                                    random_state = 255)

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
def train_step(inp, targ, enc_hidden, batch_num, val_input = val_input, val_output = val_output):
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

    if batch == len(input_tokens)//BATCH_SIZE -1 :
        print('validation loss :', loss_function(val_input, val_output))
    return batch_loss
# 모델 훈련을 위한 함수 정의

steps_per_epoch = len(input_tokens)//BATCH_SIZE

EPOCHS = 5

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

def translate(sentence):
    result, sentence = evaluate(sentence)
    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))



def validation_loss(val_input = val_input, val_output = val_output):
    val_loss = 0
    for i in range(len(val_input)):
        inputs = tf.convert_to_tensor([val_input[i]])

        result = []

        hidden = [tf.zeros((dic_input_vocab["<start>"], units))]

        enc_out, enc_hidden = encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([dic_input_vocab["<start>"]], 0)

        for t in range(output_max_len):
            predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
            
            attention_weights = tf.reshape(attention_weights, (-1,))
 
            predicted_id = tf.argmax(predictions[0]).numpy()

            result.append(predicted_id)

            dec_input = tf.expand_dims([predicted_id], 0)
            print(dec_input)
        val_loss = loss_function(val_output[i], result)

    val_loss /= len(val_input)
    return val_loss