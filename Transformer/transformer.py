import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import time
from tensorflow.keras.layers import *
from tensorflow.keras import Model
import pickle

BATCH_SIZE = 32
num_layers = 6
d_model = 512
num_heads = 8
dff = 512
Epochs = 10
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

input_vocab_size = len(input_vocab)
target_vocab_size = len(output_vocab)

input_max_len = train_input_tokens.shape[1]
output_max_len = train_output_tokens.shape[1]

def get_angles(pos, i, d_model):
    angle_rates = 1/ np.power(10000, (2 * (i // 2)) // np.float32(d_model))  # 1/ (10000의 2i/dmodel 제곱)
    return pos * angle_rates # pos와 i를 기반으로 하는 각도 제공

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], # pos
                             np.arange(d_model)[np.newaxis,:],  # i
                             d_model)
    # output shape = > (pos, i)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...] # batch 한 단어에 대한 positional encoding을 해주기 위한 값
    return tf.cast(pos_encoding, dtype = tf.float32)

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), dtype = tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :] # 어텐션 값을 추가하기 위한 차원 확장

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0) # input 에서 Lower Triangular Part만 가져옴
    return mask # (seq_len, seq_len)

def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b = True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)# scailing을 위한 차원 계산

    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)# scailing
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis = -1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0 # d_model과 num_heads는 나눠 떨어져야 연산이 가능하다.
                                             # Multi-Head Attention후 결합시에 딱 나누어 떨어져야 결합이 가능하다.
        self.depth = d_model // self.num_heads

        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)

        self.dense = Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, shape = (batch_size, -1, self.num_heads, self.depth))
        
        x = tf.transpose(x, perm = [0,2,1,3])
        # x shape : (batch_size, num_heads, seq_len, depth)

        return x

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm = [0, 2, 1, 3]) # (batch_size, seq_len, num_heads, d_model) 결합을 하기 위한 원래 형태로 변형
        
        concat_attention = tf. reshape(scaled_attention, (batch_size, -1, self.d_model)) # 분할된 head를 하나로 결합

        output = self.dense(concat_attention) # 각 단어에 대한 값을 한번더 Layer에 통과

        return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
                                Dense(dff, activation = 'relu'),
                                Dense(d_model)
    ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate = 0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = LayerNormalization(epsilon = 1e-6)
        self.layernorm2 = LayerNormalization(epsilon = 1e-6)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training = training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training = training)
        out2 = self.layernorm2(out1 +  ffn_output)

        return out2

class DecoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff, rate = 0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = LayerNormalization(epsilon = 1e-6)
        self.layernorm2 = LayerNormalization(epsilon = 1e-6)
        self.layernorm3 = LayerNormalization(epsilon = 1e-6)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)
    
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training = training)
        out1 = self.layernorm1(attn1)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training = training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training = training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2

class Encoder(Layer):
    def __init__(self, num_layers, d_model, dff, input_vocab_size, maximum_position_encoding, rate = 0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = Embedding(input_vocab_size, d_model)

        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = Dropout(rate)
    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training = training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        
        return x
        
class Decoder(Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate = 0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = Dropout(rate)
    
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        
        x = self.embedding(x)
        scailing = tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x *= scailing
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training = training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights['decoder_layer {}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer {}_block2'.format(i+1)] = block2
        return x, attention_weights

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, 
                 input_vocab_size, target_vocab_size, pe_input, pe_target, rate = 0.1):
        super(Transformer, self).__init__()
        self.encoder =  Encoder(num_layers, d_model, num_heads, input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target)

        self.final_layer = Dense(target_vocab_size)
    def call(self, inp, targ, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)

        dec_output, attention_weights = self.decoder(targ, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)

        return final_output, attention_weights

def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    
    return enc_padding_mask, combined_mask, dec_padding_mask

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction = 'none')


def loss_function(real, pred):
	mask = tf.math.logical_not(tf.math.equal(real, 0))
	loss_ = loss_object(real, pred)

	mask = tf.cast(mask, dtype=loss_.dtype)
	loss_ *= mask

	return tf.reduce_mean(loss_)

train_loss = tf.keras.metrics.Mean(name = 'train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'train_accuracy')

transformer = Transformer(num_layers,
                          d_model, 
                          num_heads,
                          dff, 
                          input_vocab_size,
                          target_vocab_size, 
                          input_vocab_size,
                          target_vocab_size)

def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
        
        loss = loss_function(tar_real, predictions)
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)

for epoch in range(Epochs):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    for batch in range(len(train_input_tokens)//BATCH_SIZE):
        batch_input = train_input_tokens[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE]
        batch_output = train_output_tokens[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE]
        train_step(batch_input, batch_output)

        if batch % 50 == 0:
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, batch, train_loss.result(), train_accuracy.result()))
    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))
