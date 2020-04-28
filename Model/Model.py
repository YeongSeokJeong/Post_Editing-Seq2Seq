from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
import tensorflow as tf

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


class Encoder_Bi_LSTM(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, lstm_hidden, batch_sz, embedding_matrix = None, multi_head_split = True):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.lstm_hidden = lstm_hidden
        if embedding_matrix is not None:
	        self.embedding = tf.keras.layers.Embedding(vocab_size, 
	                                                   embedding_dim,
	                                                   weights = [embedding_matrix],
	                                                   trainable = False
	                                                   )
        self.lstm = Bidirectional(tf.keras.layers.LSTM(self.lstm_hidden,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform',
                                       dropout = 0.3,
                                       kernel_regularizer=l2(),
                                       bias_regularizer=l2()))
		if multi_head_split == True:
	        self.multi_head_self_attention = MultiHeadAttention(lstm_hidden*2, 8)
        self.layer_norm = LayerNormalization()

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state_f, state_f_c, state_b_h, state_b_c = self.lstm(x, initial_state = hidden)
        output, state_f, state_b = self.gru(x, initial_state = hidden)

        if self.multi_head_self_attention == True:
	        output, attention_weights = self.multi_head_self_attention(output, output, output, None)

        state_h = tf.keras.layers.Concatenate()([state_f_h, state_b_h])
        state_c = tf.keras.layers.Concatenate()([state_f_c, state_b_c])
        state = [state_h, state_c]
        output = self.layer_norm(output)
        return output, state

    def initialize_hidden_state(self):
        state = tf.zeros((self.batch_sz, self.lstm_hidden))
        return [state, state, state, state]

class Encoder_Bi_GRU(tf.keras.Model):
	def __init__(self, vocab_size, embedding_dim, lstm_hidden, batch_sz, embedding_matrix = None, multi_head_split = True):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.lstm_hidden = lstm_hidden
        if embedding_matrix is not None:
	        self.embedding = tf.keras.layers.Embedding(vocab_size, 
	                                                   embedding_dim,
	                                                   weights = [embedding_matrix],
	                                                   trainable = False
	                                                   )
        self.lstm = Bidirectional(tf.keras.layers.GRU(self.lstm_hidden,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform',
                                       dropout = 0.3,
                                       kernel_regularizer=l2(),
                                       bias_regularizer=l2()))
		if multi_head_split == True:
	        self.multi_head_self_attention = MultiHeadAttention(lstm_hidden*2, 8)
        self.layer_norm = LayerNormalization()

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state_f, state_b = self.gru(x, initial_state = hidden)
        output, attention_weights = self.multi_head_self_attention(output, output, output, None)

        if self.multi_head_self_attention == True:
	        output, attention_weights = self.multi_head_self_attention(output, output, output, None)

        state = tf.keras.layers.Concatenate()([state_f, state_b])
        output = self.layer_norm(output)
        return output, state

    def initialize_hidden_state(self):
        state = tf.zeros((self.batch_sz, self.lstm_hidden))
        return [state, state]

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

class Decoder_Bi_LSTM(tf.keras.Model):
    def __init__(self,vocab_size, embedding_dim, dec_units, batch_sz, embedding_matrix = None, attention_kinds = 'BahdanauAttention'):
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
                                      recurrent_initializer='glorot_uniform',
                                      dropout = 0.3
                                      kernel_regularizer=l2(), bias_regularizer=l2())

        self.fc = tf.keras.layers.Dense(vocab_size)
        if attention_kinds == 'BahdanauAttention':
	        self.attention = BahdanauAttention(self.dec_units)
	    elif attention_kinds == 'Multihead_dot_product' :
	    	self.attention = Multihead_dot_product(self.batch_sz, dec_units, 4)
        # self.layer_norm = LayerNormalization()
        
    def call(self, x, hidden, enc_output):
        context_vector,attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector,1),x], axis = -1)
        # context_vector 와 단순 차원 합을 한다. (이어 붙임)
        output, state_h, state_c = self.lstm(x, initial_state= hidden)
        state = [state_h, state_c]
        output = tf.reshape(output,(-1,output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights

class Decoder_Bi_GRU(tf.keras.Model):
    def __init__(self,vocab_size, embedding_dim, dec_units, batch_sz, embedding_matrix = None, attention_kinds = 'BahdanauAttention'):
        # Model은 인풋과 아웃풋 텐서를 음 
        super(Decoder,self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, 
                                                   embedding_dim,
                                                   weights = [embedding_matrix],
                                                   trainable = False
                                                   )

        self.gru = tf.keras.layers.GRU(self.dec_units * 2,
                                      return_sequences=True,
                                      return_state = True,
                                      recurrent_initializer='glorot_uniform',
                                      dropout = 0.3)
        self.fc = tf.keras.layers.Dense(vocab_size)
        if attention_kinds == 'BahdanauAttention':
	        self.attention = BahdanauAttention(self.dec_units)
	    elif attention_kinds == 'Multihead_dot_product':
	    	self.attention = Multihead_dot_product(self.batch_sz, dec_units, 4)
        # self.layer_norm = LayerNormalization()
        
    def call(self, x, hidden, enc_output):
        context_vector,attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector,1),x], axis = -1)
        # context_vector 와 단순 차원 합을 한다. (이어 붙임)
        # output, state_h, state_c = self.lstm(x, initial_state= hidden)
        # state = [state_h, state_c]
        output, state = self.gru(x, initial_state = hidden)
        output = tf.reshape(output,(-1,output.shape[2]))
        output = context_vector + output
#        output += context_vector
        # output = self.layer_norm(output)
        x = self.fc(output)
        return x, state, attention_weights