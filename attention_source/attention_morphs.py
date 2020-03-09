import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import time
from tensorflow.keras.layers import *
from tensorflow.keras import Model
import pickle
from nltk.translate.bleu_score import sentence_bleu
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

BATCH_SIZE = 128
embedding_dim = 300
units = 128
morphs_train_input = []
morphs_train_output = []

morphs_val_input = []
morphs_val_output = []

morphs_test_input = []
morphs_test_output = []

with open("./data/train_input_data_1.pickle", "rb") as fr:
    train_input_tokens = pickle.load(fr)

with open("./data/train_output_data_1.pickle", 'rb') as fr:
    train_output_tokens = pickle.load(fr)

with open('./data/val_input_tokens_1.pickle', 'rb') as fr:
    val_input_tokens = pickle.load(fr)

with open('./data/val_output_tokens_1.pickle', 'rb') as fr:
    val_output_tokens = pickle.load(fr) 

with open('./data/test_input_tokens_1.pickle', 'rb') as fr:
    test_input_tokens = pickle.load(fr)

with open('./data/test_output_tokens_1.pickle', 'rb') as fr:
    test_output_tokens = pickle.load(fr)

with open('./data/input_vocab_1.pickle', 'rb') as fr:
    input_vocab = pickle.load(fr)

with open('./data/output_vocab_1.pickle', 'rb') as fr:
    output_vocab = pickle.load(fr)


dic_input_vocab = {word:i for i, word in enumerate(input_vocab)}
dic_output_vocab = {word:i for i, word in enumerate(output_vocab)}

input_max_len = train_input_tokens.shape[1]
output_max_len = train_output_tokens.shape[1]

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
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
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


strategy = tf.distribute.MirroredStrategy()


print('Number of devices : {}'.format(strategy.num_replicas_in_sync))


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

#train_input_tokens = train_input_tokens[:1000]
#train_output_tokens = train_output_tokens[:1000]

def validation_loss(val_input = val_input_tokens,  val_output = val_output_tokens):
    total_loss = 0
    input_loss = 0
    for batch in range(int(len(val_input)/BATCH_SIZE)):
        loss = 0
        inp_loss = 0
        test_input = val_input[BATCH_SIZE * batch: BATCH_SIZE  * (batch + 1)]
        test_output = val_output[BATCH_SIZE * batch: BATCH_SIZE  * (batch + 1)]
        
        inputs = tf.convert_to_tensor(test_input)
        hidden = encoder.initialize_hidden_state()

        enc_out, enc_hidden = encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([dic_input_vocab['<start>']] * BATCH_SIZE, 1)

        predict_morph = [[] for i in range(BATCH_SIZE)]

        for t in range(inputs.shape[1]-1):
            predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

            attention_weights = tf.reshape(attention_weights, (-1,))
            predictions = tf.argmax(predictions, 1)

            for i in range(BATCH_SIZE):
                predict_morph[i].append(predictions[i].numpy())

            dec_input = tf.expand_dims(predictions, 1)
        predict_morph = np.array(predict_morph)
        for i in range(BATCH_SIZE):
            inp_loss += sentence_bleu([test_input[i]], test_output[i])
            loss += sentence_bleu([predict_morph[i]], test_output[i])
        if t == 0:
            print('predict_morph :', predict_morph[0])
        loss /= BATCH_SIZE
        inp_loss /= BATCH_SIZE
        total_loss += loss
        input_loss += inp_loss
    total_loss /= int(len(val_input)/BATCH_SIZE)
    input_loss /= int(len(val_input)/BATCH_SIZE)
    return input_loss, total_loss

encoder = Encoder(len(input_vocab), embedding_dim = embedding_dim, gru_hidden = units, batch_sz = BATCH_SIZE)
decoder = Decoder(len(output_vocab), embedding_dim, units, BATCH_SIZE)

steps_per_epoch = len(train_input_tokens)//BATCH_SIZE

EPOCHS = 18

for epoch in range(EPOCHS):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for batch in range(len(train_input_tokens)//BATCH_SIZE):
        batch_input = train_input_tokens[batch*BATCH_SIZE: (batch+1)*BATCH_SIZE]
        batch_output = train_output_tokens[batch*BATCH_SIZE: (batch+1)*BATCH_SIZE]

        batch_loss = train_step(batch_input, batch_output, enc_hidden)
        total_loss += batch_loss

        if batch % 50 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))
    # saving (checkpoint) the model every 2 epochs
    inp_bleu, pred_bleu = validation_loss()
    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    print("Input sentence Bleu Score : {:.4f}".format(inp_bleu))
    print('Validation Bleu Score : {:.4f}\n'. format(pred_bleu))
    if inp_bleu < pred_bleu:
        break

def evaluate(sentence):
    inp = sentence

    inputs = tf.convert_to_tensor([inp])

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
    sen.append(input_vocab[word])
  print(sen)
  return sen

def translate(sentence):
    result, sentence = evaluate(sentence)
    kk = return_hangul(sentence)
    print('Input: ' , end = " ")
    for word in sentence:
        if word == '<p>':
            break
        print(word, end = ' ')
    print('Predicted translation: {}'.format(result))
    return sentence, result


with open("output_19Eopochs.txt", 'w') as f:
    for i in range(len(test_input_tokens)//1000):
        inp, oup = translate(test_input_tokens[i*1000])
        sen_inp = ''
        sen_cor = ''
        for word in inp:
            if word == '<p>':
                break
            sen_inp += input_vocab[word] + ' '
        for word in test_output_tokens[i*1000]:
            if word == '<p>':
                break
            else:
                sen_cor += output_vocab[word] + ' '
        f.write('input : ' + sen_inp + '\n')
        f.write('predict : ' + oup + '\n')
        f.write('correct output : ' + sen_cor + '\n')
        f.write('\n')
