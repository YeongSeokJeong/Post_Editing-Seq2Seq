import tensorflow as tf
import numpy as np
import pickle
from tokenization_morp import *
from wer import * 
import time
import os 

os.environ["CUDA_VISIBLE_DEVICES"]="0"

BATCH_SIZE = 128
units = 128 * 2

with open("./tokenized_data/train_input_data.pickle", "rb") as fr:
    train_input_token = pickle.load(fr)[:1000]

with open("./tokenized_data/train_output_data.pickle", 'rb') as fr:
    train_output_token = pickle.load(fr)[:1000]

with open('./tokenized_data/val_input_data.pickle', 'rb') as fr:
    val_input_token = pickle.load(fr)

with open('./tokenized_data/val_output_data.pickle', 'rb') as fr:
    val_output_token = pickle.load(fr) 

with open('./tokenized_data/test_input_data.pickle', 'rb') as fr:
    test_input_token = pickle.load(fr)

with open('./tokenized_data/test_output_data.pickle', 'rb') as fr:
    test_output_token = pickle.load(fr)

vocab = load_vocab('./vocab.korean_morp.list')

print('vocab size :', len(list(vocab.keys())))

ids_to_token_vocab = {i : token for i, token in enumerate(list(vocab.keys()))}

max_len = train_input_token.shape[1]

print(vocab['廃'])
# embedding_matrix = tf.train.load_variable('embedding_weight.ckpt', 'word_embeddings')
# vocab_size = embedding_matrix.shape[0]
# embedding_dim = embedding_matrix.shape[1]
# # print(vocab_size)
# # print(embedding_matrix.shape)
# # embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,
# #         				   weights = [embedding_matrix],
# #         				   trainable = False
# #         				   )
# # print(embedding(train_input_token[0]))
# tokenizer = FullTokenizer(vocab_file = './vocab.korean_morp.list')
# class Encoder(tf.keras.Model):
#     def __init__(self, vocab_size, embedding_dim, gru_hidden, batch_sz):
#         super(Encoder, self).__init__()
#         self.batch_sz = batch_sz
#         self.gru_hidden = gru_hidden
#         self.embedding = tf.keras.layers.Embedding(vocab_size, 
#         										   embedding_dim,
#         										   weights = [embedding_matrix],
#         										   trainable = False
#         										   )
#         self.gru = tf.keras.layers.GRU(self.gru_hidden,
#                                        return_sequences=True,
#                                        return_state=True,
#                                        recurrent_initializer='glorot_uniform')

#     def call(self, x, hidden):
#         x = self.embedding(x)
#         output, state = self.gru(x, initial_state = hidden)
#         return output, state

#     def initialize_hidden_state(self):
#         return tf.zeros((self.batch_sz, self.gru_hidden))

# class BahdanauAttention(tf.keras.layers.Layer):
#     def __init__(self,units):
#         super(BahdanauAttention,self).__init__()
#         self.W1 = tf.keras.layers.Dense(units)
#         self.W2 = tf.keras.layers.Dense(units)
#         self.V = tf.keras.layers.Dense(1)
#     def call(self, query,values):
#         #query => encoder hidden
#         #values => decoder hidden
#         hidden_with_time_axis = tf.expand_dims(query, 1)
#         # hidden_with_time_axis의 shape은 (batch_size, 1, hidden_size)이다.
#         score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
#         # self.W1(values) => (batch_size, seq_len, units)
#         # self.W2(hidden_with_time_axis) => (batch_size, 1, units)

#         # score shape => (batch_size, seq_len, 1)
#         attention_weights = tf.nn.softmax(score, axis = 1)
#         # softmax encoder의 단어의 중요도를 각각 얻기 위해 사용한다. 
#         context_vector = attention_weights*values
#         context_vector = tf.reduce_sum(context_vector, axis = 1)
        
#         return context_vector, attention_weights

# class Decoder(tf.keras.Model):
#     def __init__(self,vocab_size, embedding_dim, dec_units, batch_sz):
#         # Model은 인풋과 아웃풋 텐서를 음 
#         super(Decoder,self).__init__()
#         self.batch_sz = batch_sz
#         self.dec_units = dec_units
#         self.embedding = tf.keras.layers.Embedding(vocab_size, 
#         										   embedding_dim,
#         										   weights = [embedding_matrix],
#         										   trainable = False
#         										   )
#         self.gru = tf.keras.layers.GRU(self.dec_units,
#                                       return_sequences=True,
#                                       return_state = True,
#                                       recurrent_initializer='glorot_uniform')
#         self.fc = tf.keras.layers.Dense(vocab_size)
#         self.attention = BahdanauAttention(self.dec_units)
        
#     def call(self,x,hidden, enc_output):
#         context_vector,attention_weights = self.attention(hidden, enc_output)
#         x = self.embedding(x)
#         x = tf.concat([tf.expand_dims(context_vector,1),x], axis = -1)
#         # context_vector 와 단순 차원 합을 한다. (이어 붙임)
#         output,state = self.gru(x)
#         output = tf.reshape(output,(-1,output.shape[2]))
#         x = self.fc(output)
#         return x, state, attention_weights

# optimizer = tf.keras.optimizers.Adam()
# # 최적화 함수 정의
# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction = 'none')

# def loss_function(real,pred):
#     mask = tf.math.logical_not(tf.math.equal(real,0))
#     loss_ = loss_object(real,pred)
    
#     mask = tf.cast(mask,dtype = loss_.dtype)
#     loss_ *= mask
#     return tf.reduce_mean(loss_)
# # 손실함수 정의

# def train_step(inp, targ, enc_hidden):
#     loss = 0

#     with tf.GradientTape() as tape:
#         enc_output, enc_hidden = encoder(inp, enc_hidden)

#         dec_hidden = enc_hidden

#         dec_input = tf.expand_dims([vocab['[CLS]']] * BATCH_SIZE, 1)

#         for t in range(1, targ.shape[1]):
#             predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

#             loss += loss_function(targ[:, t], predictions)

#             dec_input = tf.expand_dims(targ[:, t], 1)

#     batch_loss = (loss / int(targ.shape[1]))

#     variables = encoder.trainable_variables + decoder.trainable_variables

#     gradients = tape.gradient(loss, variables)

#     optimizer.apply_gradients(zip(gradients, variables))

#     return batch_loss

# def validation_loss(val_input = val_input_token,  val_output = val_output_token):
#     total_loss = 0
#     input_loss = 0
#     for batch in range(int(len(val_input)/BATCH_SIZE)):
#         loss = 0
#         inp_loss = 0
#         test_input = val_input[BATCH_SIZE * batch: BATCH_SIZE  * (batch + 1)]
#         test_output = val_output[BATCH_SIZE * batch: BATCH_SIZE  * (batch + 1)]
        
#         inputs = tf.convert_to_tensor(test_input)
#         hidden = encoder.initialize_hidden_state()

#         enc_out, enc_hidden = encoder(inputs, hidden)

#         dec_hidden = enc_hidden
#         dec_input = tf.expand_dims([vocab['[CLS]']] * BATCH_SIZE, 1)

#         predict_morph = [[] for i in range(BATCH_SIZE)]

#         for t in range(inputs.shape[1]-1):
#             predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

#             attention_weights = tf.reshape(attention_weights, (-1,))
#             predictions = tf.argmax(predictions, 1)

#             for i in range(BATCH_SIZE):
#                 predict_morph[i].append(predictions[i].numpy())
#             dec_input = tf.expand_dims(predictions, 1)
        
#         test_inp = []
#         test_targ = []
#         predic = []
        
#         print(test_input[i])
        
#         for i in range(BATCH_SIZE):
#             step_input = [ids_to_token_vocab[word] for word in test_input[i] if word not in [0, 1, 2]]
#             step_output = [ids_to_token_vocab[word] for word in test_output[i] if word not in [0, 1, 2]]
#             step_predict = [ids_to_token_vocab[word] for word in predict_morph[i] if word not in [0, 1, 2]]
#             test_inp.append(step_input)
#             test_targ.append(step_output)
#             predic.append(step_predict)

#         for i in range(BATCH_SIZE):
#             inp_loss += wer(test_targ[i], test_inp[i])
#             loss += wer(test_targ[i], predic[i])
#         loss /= BATCH_SIZE
#         inp_loss /= BATCH_SIZE
#         total_loss += loss
#         input_loss += inp_loss
#     total_loss /= int(len(val_input)/BATCH_SIZE)
#     input_loss /= int(len(val_input)/BATCH_SIZE)
#     return input_loss, total_loss

# encoder = Encoder(vocab_size, embedding_dim = embedding_dim, gru_hidden = units, batch_sz = BATCH_SIZE)
# decoder = Decoder(vocab_size, embedding_dim, units, BATCH_SIZE)

# steps_per_epoch = len(train_input_token)//BATCH_SIZE

# EPOCHS = 1

# for epoch in range(EPOCHS):
#     start = time.time()

#     enc_hidden = encoder.initialize_hidden_state()
#     total_loss = 0

#     for batch in range(len(train_input_token)//BATCH_SIZE):
#         batch_input = train_input_token[batch*BATCH_SIZE: (batch+1)*BATCH_SIZE]
#         batch_output = train_output_token[batch*BATCH_SIZE: (batch+1)*BATCH_SIZE]

#         batch_loss = train_step(batch_input, batch_output, enc_hidden)
#         total_loss += batch_loss

#         if batch % 50 == 0:
#             print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
#                                                          batch,
#                                                          batch_loss.numpy()))
#     # saving (checkpoint) the model every 2 epochs
#     inp_wer, pred_wer = validation_loss()
#     print('Epoch {} Loss {:.4f}'.format(epoch + 1,
#                                       total_loss / steps_per_epoch))
#     print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
#     print("Input sentence Bleu Score : {:.4f}%".format(inp_bleu))
#     print('Validation Bleu Score : {:.4f}%\n'. format(pred_bleu))

# def evaluate(sentence):
#     inp = sentence

#     inputs = tf.convert_to_tensor([inp])

#     result = ''
    
#     hidden = [tf.zeros((1, units))]
#     enc_out, enc_hidden = encoder(inputs, hidden)

#     dec_hidden = enc_hidden
#     dec_input = tf.expand_dims([vocab['[CLS]']], 0)

#     for t in range(output_max_len):
#         predictions, dec_hidden, attention_weights = decoder(dec_input, 
#                                                              dec_hidden,
#                                                              enc_out)
#         attention_weights = tf.reshape(attention_weights, (-1,))
        

#         predicted_id = tf.argmax(predictions[0]).numpy()

#         result += ids_to_token_vocab[predicted_id] + ' '
        
#         if output_vocab[predicted_id] == '[SEP]':
#             return result, sentence
#         dec_input = tf.expand_dims([predicted_id], 0)

#     return result, sentence

# def translate(sentence):
#     result, sentence = evaluate(sentence)
#     kk = tokenizer.convert_ids_to_tokens(ids_to_token_vocab, sentence)
#     print('Input: ' , end = " ")
#     for word in sentence:
#         if word == '[PAD]':
#             break
#         print(word, end = ' ')
#     print('Predicted translation: {}'.format(result))
#     return sentence, result


# with open("output_18Eopochs.txt", 'w') as f:
#     for i in range(len(test_input_tokens)):
#         inp, oup = translate(test_input_tokens[i])
#         sen_inp = ''
#         sen_cor = ''
#         for word in inp:
#             if word == '[PAD]':
#                 break
#             sen_inp += ids_to_token_vocab[word] + ' '
#         for word in test_output_tokens[i]:
#             if word == '[PAD]':
#                 break
#             else:
#                 sen_cor += ids_to_token_vocab[word] + ' '
#         f.write('input : ' + sen_inp + '\n')
#         f.write('predict : ' + oup + '\n')
#         f.write('correct output : ' + sen_cor + '\n')
#         f.write('\n')