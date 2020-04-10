import tensorflow as tf
import numpy as np
import pickle
from tokenization_morp import *
from wer import * 
import time
import os 
from tensorflow.keras.layers import *
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"]="0"

BATCH_SIZE = 128*3
units = 128 * 2

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

ids_to_token_vocab = {i : token for i, token in enumerate(keys)}

max_len = train_input_token.shape[1]


embedding_matrix = tf.train.load_variable('embedding_weight.ckpt', 'word_embeddings')
vocab_size = embedding_matrix.shape[0]
embedding_dim = embedding_matrix.shape[1]

tokenizer = FullTokenizer(vocab_file = './vocab.korean_morp.list')

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, lstm_hidden, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.lstm_hidden = lstm_hidden
        self.embedding = tf.keras.layers.Embedding(vocab_size, 
                                                   embedding_dim,
                                                   weights = [embedding_matrix],
                                                   trainable = False
                                                   )
        self.lstm = Bidirectional(tf.keras.layers.LSTM(self.lstm_hidden,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform'))


    def call(self, x, hidden):
        x = self.embedding(x)
        output, state_f_h, state_f_c, state_b_h, state_b_c = self.lstm(x, initial_state = hidden)
        state_h = tf.keras.layers.Concatenate()([state_f_h, state_b_h])
        state_c = tf.keras.layers.Concatenate()([state_f_c, state_b_c])
        state = [state_h, state_c]
        return output, state

    def initialize_hidden_state(self):
        state = tf.zeros((self.batch_sz, self.lstm_hidden))
        return [state, state, state, state]

class MultiHead_dot_product_Attention(tf.keras.layers.Layer):
    def __init__(self, batch_size, units, head_num):
        super(MultiHead_dot_product_Attention, self).__init__()
        self.head_num = head_num
        self.units = units
        self.split_head_units = int(units/head_num) * 2
        self.batch_size = batch_size
        
        if divmod(units,head_num)[1] != 0:
            raise ValueError("please check unit size and head size")
    
    def split_head(self, inp):
        return tf.reshape(inp, (inp.shape[0], self.head_num, -1, self.split_head_units))
        
    def call(self, query, values):
        query = tf.expand_dims(query, axis = 1)
        split_query = self.split_head(query)
        split_values = self.split_head(values)
        attention_weights = tf.linalg.matmul(split_query, split_values, transpose_b = True)
        attention_weights = tf.nn.softmax(attention_weights)
        attention_weights = tf.reshape(attention_weights, (query.shape[0], self.head_num, -1, 1))
        
        attention_score = split_values * attention_weights
        attention_score = tf.reshape(attention_score, shape = (query.shape[0], -1, self.units))
        attention_score = tf.reduce_sum(attention_score, axis = 1)
        return attention_score, attention_weights

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
        self.lstm = tf.keras.layers.LSTM(self.dec_units * 2,
                                      return_sequences=True,
                                      return_state = True,
                                      recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = MultiHead_dot_product_Attention(self.batch_sz, units, 4)
        
    def call(self, x, hidden, enc_output):
        context_vector,attention_weights = self.attention(hidden[0], enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector,1),x], axis = -1)
        # context_vector 와 단순 차원 합을 한다. (이어 붙임)
        output, state_h, state_c = self.lstm(x, initial_state= hidden)
        state = [state_h, state_c]
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

def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([vocab['[CLS]']] * BATCH_SIZE, 1)

        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)

            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

def make_sentence(sentences):
    return [[ids_to_token_vocab[word] for word in sentence if word not in [0,1,2,3]] for sentence in sentences]

def validation_loss(val_input = val_input_token,  val_output = val_output_token):
    total_loss = 0
    for batch in range(int(len(val_input)/BATCH_SIZE)):
        loss = 0
        test_input = val_input[BATCH_SIZE * batch: BATCH_SIZE  * (batch + 1)]
        test_output = val_output[BATCH_SIZE * batch: BATCH_SIZE  * (batch + 1)]
        
        inputs = tf.convert_to_tensor(test_input)
        hidden = encoder.initialize_hidden_state()

        enc_out, enc_hidden = encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([vocab['[CLS]']] * BATCH_SIZE, 1)

        predict_morph = [[] for i in range(BATCH_SIZE)]

        for t in range(inputs.shape[1]-1):
            predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

            attention_weights = tf.reshape(attention_weights, (-1,))
            predictions = tf.argmax(predictions, 1)

            for i in range(BATCH_SIZE):
                predict_morph[i].append(predictions[i].numpy())
            dec_input = tf.expand_dims(predictions, 1)
        
        test_targ = []
        predic = []
                        
        test_targ = make_sentence(test_output)
        predic = make_sentence(predict_morph)
        if batch == 0:
            print(predic[0])
        for i in range(BATCH_SIZE):
            loss += wer(test_targ[i], predic[i])
        loss /= BATCH_SIZE
        total_loss += loss
    total_loss /= int(len(val_input)/BATCH_SIZE)
    return total_loss
encoder = Encoder(vocab_size, embedding_dim = embedding_dim, lstm_hidden = units, batch_sz = BATCH_SIZE)
decoder = Decoder(vocab_size, embedding_dim, units, BATCH_SIZE)

steps_per_epoch = len(train_input_token)//BATCH_SIZE

EPOCHS = 8

for epoch in tqdm(range(EPOCHS)):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for batch in tqdm(range(len(train_input_token)//BATCH_SIZE)):
        batch_input = train_input_token[batch*BATCH_SIZE: (batch+1)*BATCH_SIZE]
        batch_output = train_output_token[batch*BATCH_SIZE: (batch+1)*BATCH_SIZE]

        batch_loss = train_step(batch_input, batch_output, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))
    # saving (checkpoint) the model every 2 epochs
    pred_wer = validation_loss()
    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    print('Validation Bleu Score : {:.4f}%\n'. format(pred_wer))

def evaluate(sentence):
    inp = sentence

    inputs = tf.convert_to_tensor([inp])

    result = ''
    
    hidden = [tf.zeros((1, units)), tf.zeros(1, units), tf.zeros(1, units), tf.zeros(1, units)]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([vocab['[CLS]']], 0)

    for t in range(max_len):
        predictions, dec_hidden, attention_weights = decoder(dec_input, 
                                                             dec_hidden,
                                                             enc_out)
        attention_weights = tf.reshape(attention_weights, (-1,))
        

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += ids_to_token_vocab[predicted_id] + ' '
        
        if ids_to_token_vocab[predicted_id] ==  '[SEP]':
            return result
        dec_input = tf.expand_dims([predicted_id], 0)

    return result

def translate(sentence):
    result = evaluate(sentence)
    return result

output_file_name = "attention_bert" + str(EPOCHS) + ".txt"
with open(output_file_name, 'w') as f:
    for i in range(len(test_input_token)):
        oup = translate(test_input_token[i])
        sen_inp = ''
        sen_cor = ''
        for word in test_input_token[i]:
            if word not in [0,1,2,3]:
                sen_inp += ids_to_token_vocab[word] + ' '
            
        for word in test_output_token[i]:
            if word not in [0,1,2,3]:
                sen_cor += ids_to_token_vocab[word] + ' '
        f.write('input : ' + sen_inp + '\n')
        f.write('predict : ' + oup + '\n')
        f.write('correct output : ' + sen_cor + '\n')
        f.write('\n')

print(output_file_name)