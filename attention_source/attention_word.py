import pandas as pandas
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import  Model
import pickle
import time
import nltk.translate.bleu_score as bleu
BATCH_SIZE = 128

embedding_dim = 300
units = 128

# words_train_input = []
# words_train_output = []

# words_val_input = []
# words_val_output = []

# words_test_input = []
# words_test_output = []

# words_input_vocab = []
# words_output_vocab = []

with open('./data/train_word_input_data.pickle', 'rb') as fr:
	words_train_input = pickle.load(fr)

with open('./data/train_word_output_data.pickle', 'rb') as fr:
	words_train_output = pickle.load(fr)

with open('./data/val_word_input_tokens.pickle', 'rb') as fr:
	words_val_input = pickle.load(fr)

with open('./data/val_word_output_tokens.pickle', 'rb') as fr:
	words_val_output = pickle.load(fr)

with open('./data/test_word_input_tokens.pickle', 'rb') as fr:
	words_test_input = pickle.load(fr)

with open('./data/test_word_output_tokens.pickle', 'rb') as fr:
	words_test_output = pickle.load(fr)

with open('./data/word_input_vocab.pickle', 'rb') as fr:
	words_input_vocab = pickle.load(fr)

with open('./data/word_output_vocab.pickle', 'rb') as fr:
	words_output_vocab = pickle.load(fr)

dic_input_vocab = {word:i for i, word in enumerate(words_input_vocab)}
dic_output_vocab = {word:i for i, word in enumerate(words_output_vocab)}
input_max_len = words_train_input.shape[1]
output_max_len = words_train_output.shape[1]

class Encoder(tf.keras.Model):
	def __init__(self, vocab_size, embedding_dim, gru_hidden, batch_sz):
		super(Encoder, self).__init__()
		self.batch_sz = batch_sz
		self.gru_hidden = gru_hidden
		self.embedding = Embedding(vocab_size, embedding_dim)
		self.gru = GRU(self.gru_hidden,
					   return_sequences = True,
					   return_state = True,
					   recurrent_initializer = 'glorot_uniform')

	def call(self, x, hidden):
		x = self.embedding(x)
		output, state = self.gru(x, initial_state = hidden)
		return output, state

	def initialize_hidden_state(self):
		return tf.zeros((self.batch_sz, self.gru_hidden))

class BahdanauAttention(tf.keras.layers.Layer):
	def __init__(self, units):
		super(BahdanauAttention, self).__init__()
		self.W1 = Dense(units)
		self.W2 = Dense(units)
		self.V = Dense(1)

	def call(self, query, values):
		hidden_with_time_axis = tf.expand_dims(query, 1)
		score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
		
		attention_weights = tf.nn.softmax(score, axis = 1)

		context_vector = attention_weights * values
		context_vector = tf.reduce_sum(context_vector, axis = 1)

		return context_vector, attention_weights

class Decoder(tf.keras.Model):
	def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
		super(Decoder, self).__init__()
		self.batch_sz = batch_sz
		self.dec_units = dec_units
		self.embedding = Embedding(vocab_size,
								   embedding_dim)
		self.gru = GRU(self.dec_units,
					   return_sequences = True,
					   return_state = True,
					   recurrent_initializer = 'glorot_uniform')
		self.fc = Dense(vocab_size)
		self.attention = BahdanauAttention(self.dec_units)

	def call(self, x, hidden, enc_output):
		context_vector, attention_weights = self.attention(hidden, enc_output)
		
		x = self.embedding(x)
		x = tf.concat([tf.expand_dims(context_vector, 1), x], axis = -1)
		
		output, state = self.gru(x)
		output = tf.reshape(output, (-1, output.shape[2]))
		
		x = self.fc(output)
		
		return x, state, attention_weights

optimizer = tf.keras.optimizers.Adam()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True,
															reduction = 'none')

def loss_function(real, pred):
	mask = tf.math.logical_not(tf.math.equal(real, 0))
	loss_ = loss_object(real, pred)

	mask = tf.cast(mask, dtype = loss_.dtype)
	loss_ *= mask
	return tf.reduce_mean(loss_)

def train_step(inp, targ, enc_hidden):
	loss = 0

	with tf.GradientTape() as tape:
		enc_output, enc_hidden = encoder(inp, enc_hidden)

		dec_hidden = enc_hidden

		dec_input = tf.expand_dims([dic_output_vocab['<start>']] * BATCH_SIZE, 1)

		for t in range(1, targ.shape[1]):
			predictions, dec_hidden, _  = decoder(dec_input, dec_hidden, enc_output)
			loss += loss_function(targ[:, t], predictions)

	batch_loss = (loss / int(targ.shape[1]))

	variables = encoder.trainable_variables + decoder.trainable_variables

	gradients = tape.gradient(loss, variables)

	optimizer.apply_gradients(zip(gradients, variables))

	return batch_loss

def validation_loss(val_input = words_val_input, val_output = words_val_output):
	total_loss = 0

	for batch in range(int(len(val_input)/BATCH_SIZE)):
		loss = 0

		test_input = val_input[BATCH_SIZE * batch: BATCH_SIZE  * (batch + 1)]
		test_output = val_output[BATCH_SIZE * batch: BATCH_SIZE  * (batch + 1)]

		inputs = tf.convert_to_tensor(test_input)
		hidden = encoder.initialize_hidden_state()

		enc_out, enc_hidden = encoder(inputs, hidden)
		dec_hidden = enc_hidden
		dec_input = tf.expand_dims([dic_input_vocab['<start>']] * BATCH_SIZE, 1)

		predict = [[] for i in range(128)]

		for t in range(inputs.shape[1]):
		    predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

		    attention_weights = tf.reshape(attention_weights, (-1,))

		    predictions = tf.argmax(predictions, 1)
		    for i in range(BATCH_SIZE):
		    	predict[i].append(predictions[i])
		    dec_input = tf.expand_dims(predictions, 1)

	    for i in range(BATCH_SIZE):
	    	loss += bleu(predict[i], test_output[i])
	    
		loss = loss/BATCH_SIZE
        
		total_loss += loss
	total_loss /= int(len(val_input)/BATCH_SIZE)
	return total_loss


encoder = Encoder(len(words_input_vocab), embedding_dim, units, BATCH_SIZE)
decoder = Decoder(len(words_output_vocab), embedding_dim, units, BATCH_SIZE)

steps_per_epoch = len(words_train_input)//BATCH_SIZE

EPOCHS = 3

for epoch in range(EPOCHS):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for batch in range(len(words_train_input)//BATCH_SIZE):
        batch_input = words_train_input[batch*BATCH_SIZE: (batch+1)*BATCH_SIZE]
        batch_output = words_train_output[batch*BATCH_SIZE: (batch+1)*BATCH_SIZE]
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

with open("./result/word_output_3Eopochs.txt", 'w') as f:
    for i in range(len(test_input_tokens)//1000):
        inp, oup = translate(test_input_tokens[i*1000])
        sen_inp = ''
        for word in inp:
            if word == '<p>':
                break
            sen_inp += input_vocab[word]
        f.write('input : ' + sen_inp + '\n')
        f.write('predict : ' + oup + '\n')