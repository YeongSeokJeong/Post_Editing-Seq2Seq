import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import time
from tensorflow.keras.layers import *
from tensorflow.keras import Model
import pickle
from nltk.translate.bleu_score import sentence_bleu

BATCH_SIZE = 128
embedding_dim = 300
units = 128
morphs_train_input = []
morphs_train_output = []

morphs_val_input = []
morphs_val_output = []

morphs_test_input = []
morphs_test_output = []

with open("./data/train_input_data.pickle", "rb") as fr:
    train_input_tokens = pickle.load(fr)

with open("./data/train_output_data.pickle", 'rb') as fr:
    train_output_tokens = pickle.load(fr)

with open('./data/val_input_tokens.pickle', 'rb') as fr:
    val_input_tokens = pickle.load(fr)

with open('./data/val_output_tokens.pickle', 'rb') as fr:
    val_output_tokens = pickle.load(fr) 

with open('./data/test_input_tokens.pickle', 'rb') as fr:
    test_input_tokens = pickle.load(fr)

with open('./data/test_output_tokens.pickle', 'rb') as fr:
    test_output_tokens = pickle.load(fr)

with open('./data/input_vocab.pickle', 'rb') as fr:
    input_vocab = pickle.load(fr)

with open('./data/output_vocab.pickle', 'rb') as fr:
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

encoder = Encoder(len(input_vocab), embedding_dim = embedding_dim, gru_hidden = units, batch_sz = BATCH_SIZE)
decoder = Decoder(len(output_vocab), embedding_dim, units, BATCH_SIZE)

strategy = tf.distribute.MirroredStrategy()

print('Number of devices : {}'.format(strategy.num_replicas_in_sync))

buffer_size = len(train_input_tokens)

Batch_size_per_replica = BATCH_SIZE * strategy.num_replicas_in_sync
Global_batch_size = Batch_size_per_replica * strategy.num_replicas_in_sync
print("global batch size : " Global_batch_size)
train_dataset = tf.data.Dataset.from_tensor_slices((train_input_tokens, train_output_tokens)).shuffle(buffer_size).batch(Global_batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_input_tokens, test_output_tokens)).batch(Global_batch_size)

train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

with strategy.scope():
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction = 'none')
    optimizer = tf.keras.optimizers.Adam()
    def loss_function(real,pred):
        mask = tf.math.logical_not(tf.math.equal(real,0))
        loss_ = loss_object(real,pred)
        
        mask = tf.cast(mask,dtype = loss_.dtype)
        loss_ *= mask
        return loss_

    def compute_loss(labels, predictions):
        per_example_loss = loss_function(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size = Global_batch_size)
    

with strategy.scope():
    test_loss = tf.keras.metrics.Mean(name = 'test_loss')

    train_accuracy = tf.keras.metrics.SparseCategoricalCrossentropy(name = 'train_accuracy')
    test_accuracy = tf.keras.metrics.SparseCategoricalCrossentropy(name = 'test_accuracy')

with strategy.scope():
    def train_step(inputs, ):
        inp, targ = inputs
        loss = 0
        enc_hidden = encoder.initialize_hidden_state()
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(inp, enc_hidden)

            dec_hidden = enc_hidden

            dec_input = tf.expand_dims([dic_output_vocab['<start>']] * BATCH_SIZE, 1)

            for t in range(1, targ.shape[1]):
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                loss += compute_loss([targ[:, t]], predictions)

                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))

        variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss

Epochs = 10

with strategy.scope():
    def distributed_train_step(dataset_inputs, enc_hidden):
        per_replica_losses = strategy.experimental_run_v2(train_step,
                                                        args = (dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis = None)

    for epoch in range(Epochs):
        start = time.time()
        enc_hidden = encoder.initialize_hidden_state()

        total_loss = 0
        num_batchs = 0
        for x in train_dist_dataset:
            total_loss = distributed_train_step(x, enc_hidden)
            num_batchs += 1

            if num_batchs % 10 == 0:
                print('batch : {} current train set loss : {:.4f}'.format(num_batchs, total_loss.numpy()))
        train_loss = total_loss / num_batchs

        print('Epoch {} Loss {}'.format(epoch + 1, train_loss))
