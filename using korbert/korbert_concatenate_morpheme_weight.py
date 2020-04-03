import tensorflow as tf
import numpy as np
import pickle
from tokenization_morp import *
from wer import * 
import time
import os 
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"]="0"

BATCH_SIZE = 64
units = 128 * 3

with open("./tokenized_data/train_input_data.pickle", "rb") as fr:
    train_input_token = pickle.load(fr)

with open("./tokenized_data/train_output_data.pickle", 'rb') as fr:
    train_output_token = pickle.load(fr)

with open('./tokenized_data/val_input_data.pickle', 'rb') as fr:
    val_input_token = pickle.load(fr)

with open('./tokenized_data/val_output_data.pickle', 'rb') as fr:
    val_output_token = pickle.load(fr) 

with open('./tokenized_data/test_input_data.pickle', 'rb') as fr:
    test_input_token = pickle.load(fr)

with open('./tokenized_data/test_output_data.pickle', 'rb') as fr:
    test_output_token = pickle.load(fr)

vocab = load_vocab('./vocab.korean_morp.list')

keys = vocab.keys()

print()
print("len key:",len(keys))
print()
ids_to_token_vocab = {i : token for i, token in enumerate(keys)}

max_len = train_input_token.shape[1]


embedding_matrix = tf.train.load_variable('embedding_weight.ckpt', 'word_embeddings')
vocab_size = embedding_matrix.shape[0]
embedding_dim = embedding_matrix.shape[1]

with open("./tokenized_data/train_input_data.pickle", "rb") as fr:
    train_input_morph = pickle.load(fr)

with open("./tokenized_data/train_output_data.pickle", 'rb') as fr:
    train_output_morph = pickle.load(fr)

with open('./tokenized_data/val_input_data.pickle', 'rb') as fr:
    val_input_morph = pickle.load(fr)

with open('./tokenized_data/val_output_data.pickle', 'rb') as fr:
    val_output_morph = pickle.load(fr) 

with open('./tokenized_data/test_input_data.pickle', 'rb') as fr:
    test_input_morph = pickle.load(fr)

with open('./tokenized_data/test_output_data.pickle', 'rb') as fr:
    test_output_morph = pickle.load(fr)
    
with open('./tokenized_data/input_vocab.pickle', 'rb') as fr:
    input_vocab = pickle.load(fr)
    
with open('./tokenized_data/output_vocab.pickle', 'rb') as fr:
    output_vocab = pickle.load(fr)

class Encoder(tf.keras.Model):
    def __init__(self, bert_vocab_size, morpheme_vocab_size , embedding_dim, gru_hidden, batch_sz, max_seq_len):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.gru_hidden = gru_hidden
        self.embedding_bert = tf.keras.layers.Embedding(vocab_size, 
        										   embedding_dim,
        										   weights = [embedding_matrix],
        										   trainable = False
        										   )
        self.embedding_morpheme = tf.keras.layers.Embedding(morpheme_vocab_size, embedding_dim)
        self.cnn1d = tf.keras.layers.Conv1D(max_seq_len,3)
        self.max_pooling = tf.keras.layers.GlobalMaxPool1D()
        
        self.lstm = tf.keras.layers.LSTM(embedding_dim,
                                        return_sequences = True,
                                        recurrent_initializer='glorot_uniform')
        
        self.gru = tf.keras.layers.GRU(self.gru_hidden,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
    def call(self, x, morpheme, hidden):
        x = self.embedding_bert(x)
        morpheme = self.embedding_morpheme(morpheme)
        morpheme = self.cnn1d(morpheme)
        morpheme = self.max_pooling(morpheme)
        morpheme = tf.expand_dims(morpheme, axis = -1)
        x = tf.concat([x, morpheme], -1)
        lstm_output = self.lstm(x)        
        output, state = self.gru(lstm_output, initial_state = hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.gru_hidden))

encoder = Encoder(len(ids_to_token_vocab), 
                  len(input_vocab), 
                  embedding_dim,
                  units,
                  BATCH_SIZE, 
                  train_input_token.shape[1])
sample_input = train_input_token[:BATCH_SIZE]
sample_morpheme = train_input_morph[:BATCH_SIZE]
sample_output = encoder(sample_input, sample_morpheme, encoder.initialize_hidden_state())

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
        self.embedding = tf.keras.layers.Embedding(vocab_size, 
        										   embedding_dim,
        										   weights = [embedding_matrix],
        										   trainable = False
        										   )
        
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                      return_sequences=True,
                                      return_state = True,
                                      recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)
    
    def call(self,x, hidden, enc_output):
        context_vector,attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector,1),x], axis = -1)
        # context_vector 와 단순 차원 합을 한다. (이어 붙임)
        output,state = self.gru(x)
        output = tf.reshape(output,(-1,output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

def train_step(inp, inp_morpheme, targ, hidden):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, inp_morpheme, hidden)
        dec_input = tf.expand_dims([1]* BATCH_SIZE, 1)

        dec_hidden = enc_hidden
        for t in range(1, targ.shape[1]):
            predictions,dec_hidden, _ = decoder(dec_input, enc_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t], 1)
    batch_loss = (loss/int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    
    return batch_loss

encoder = Encoder(len(ids_to_token_vocab), 
                  len(input_vocab), 
                  embedding_dim,
                  units,
                  BATCH_SIZE, 
                  train_input_token.shape[1])
decoder = Decoder(len(ids_to_token_vocab), embedding_dim, units, BATCH_SIZE)

EPOCHS = 20
steps_per_epoch = len(train_input_token) // BATCH_SIZE
for epoch in range(EPOCHS):
    start = time.time()
    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0
    for batch in range(len(train_input_token)//BATCH_SIZE):
        batch_input = train_input_token[batch*BATCH_SIZE: (batch+1)*BATCH_SIZE]
        batch_morpheme_input = train_input_morph[batch*BATCH_SIZE : (batch+1)*BATCH_SIZE]
        batch_output = train_output_token[batch*BATCH_SIZE: (batch+1)*BATCH_SIZE]

        batch_loss = train_step(batch_input, batch_morpheme_input, batch_output, enc_hidden)
        
        total_loss += batch_loss
        if batch % 50 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))
    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))