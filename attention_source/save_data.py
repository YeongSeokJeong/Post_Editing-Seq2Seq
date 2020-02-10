import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from konlpy.tag import Kkma
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

train_input, test_input, train_output, test_output = train_test_split(input_data,
                                                                      output_data,
                                                                      test_size = 0.2)

train_input, val_input, train_output, val_output = train_test_split(train_input,
                                                                    train_output,
                                                                    test_size = 0.125)
print("original size")
print('train size : {}'.format(len(train_input)))
print('val size : {}'.format(len(val_input)))
print('test size : {}\n'.format(len(test_input)))

train_input, train_output = make_char_data(train_input, 
                                           train_output, 
                                           int(len(train_input)/2))

print("Adding char modified data size")
print('train size : {}'.format(len(train_input)))
print('val size : {}'.format(len(val_input)))
print('test size : {}\n'.format(len(test_input)))

for i,(tr_input, tr_output) in enumerate(zip(train_input, train_output)):
    input_sentence = kkma.morphs(tr_input)
    output_sentence = kkma.morphs(tr_output)
    
    morphs_train_input.append(input_sentence)
    morphs_train_output.append(output_sentence)
    
    input_vocab.update(input_sentence)
    output_vocab.update(output_sentence)
    
    input_steplen = len(input_sentence)
    output_steplen = len(output_sentence)
    
    input_max_len = input_max_len if input_max_len > input_steplen else input_steplen
    output_max_len = output_max_len if output_max_len > output_steplen else output_steplen

    if i%5000 == 0 :
        print('{}/{} complete train_morph!'.format(i, len(train_input)))
print(len(input_vocab))

for i,(v_input, v_output) in enumerate(zip(val_input, val_output)):
    input_sentence = kkma.morphs(v_input)
    output_sentence = kkma.morphs(v_output)
    
    morphs_val_input.append(input_sentence)
    morphs_val_output.append(output_sentence)
    
    input_vocab.update(input_sentence)
    output_vocab.update(output_sentence)
    
    input_steplen = len(input_sentence)
    output_steplen = len(output_sentence)
    
    input_max_len = input_max_len if input_max_len > input_steplen else input_steplen
    output_max_len = output_max_len if output_max_len > output_steplen else output_steplen


    if i%5000 == 0 :
        print('{}/{} complete val_morph!'.format(i,len(val_input)))
print(len(input_vocab))

for i,(te_input, te_output) in enumerate(zip(test_input, test_output)):
    input_sentence = kkma.morphs(te_input)
    output_sentence = kkma.morphs(te_output)
    
    morphs_test_input.append(input_sentence)
    morphs_test_output.append(output_sentence)
    
    input_vocab.update(input_sentence)
    output_vocab.update(output_sentence)
    
    input_steplen = len(input_sentence)
    output_steplen = len(output_sentence)
    
    input_max_len = input_max_len if input_max_len > input_steplen else input_steplen
    output_max_len = output_max_len if output_max_len > output_steplen else output_steplen
    
    if i%5000 == 0 :
        print('{}/{} complete test_morph!'.format(i, len(test_input)))
print(len(input_vocab))


morphs_train_input, morphs_train_output = make_morph_data(morphs_train_input,
                                                          morphs_train_output, 
                                                          input_vocab, 
                                                          len(morphs_train_input)//2)

print('Adding morphs modified data size')
print('train size : {}'.format(len(morphs_train_input)))
print('val size : {}'.format(len(morphs_val_input)))
print('test size : {}\n'.format(len(morphs_test_input)))


input_vocab = ['<p>', '<start>', '<end>'] + list(input_vocab)
output_vocab = ['<p>','<start>', '<end>'] + list(output_vocab)
# vocab에 패딩 단어, 시작 단어, 끝 단어를 추가한다.

train_input_tokens = []
train_output_tokens = []

val_input_tokens = []
val_output_tokens = []

test_input_tokens = []
test_output_tokens = []

dic_input_vocab = {word : i for i, word in enumerate(input_vocab)}
dic_output_vocab = {word: i for i, word in enumerate(output_vocab)}


for i, (v_input, v_output) in enumerate(zip(morphs_val_input, morphs_val_output)):
    input_steplen = len(v_input)
    output_steplen = len(v_output)

    step_input = [dic_input_vocab["<start>"]] + [dic_input_vocab[word] for word in v_input] + [dic_input_vocab["<end>"]]
    step_output =  [dic_output_vocab["<start>"]] + [dic_output_vocab[word] for word in v_output] + [dic_output_vocab["<end>"]]
    
    val_input_tokens.append(step_input)
    val_output_tokens.append(step_output)

    input_max_len = input_max_len if input_max_len > input_steplen else input_steplen
    output_max_len = output_max_len if output_max_len > output_steplen else output_steplen

for i, (t_input, t_output) in enumerate(zip(morphs_test_input, morphs_test_output)):
    input_steplen = len(t_input)
    output_steplen = len(t_output)

    step_input = [dic_input_vocab["<start>"]] + [dic_input_vocab[word] for word in t_input] + [dic_input_vocab["<end>"]]
    step_output =  [dic_output_vocab["<start>"]] + [dic_output_vocab[word] for word in t_output] + [dic_output_vocab["<end>"]]
    
    test_input_tokens.append(step_input)
    test_output_tokens.append(step_output)

    input_max_len = input_max_len if input_max_len > input_steplen else input_steplen
    output_max_len = output_max_len if output_max_len > output_steplen else output_steplen

for i, (tr_input, tr_output) in enumerate(zip(morphs_train_input, morphs_train_output)):
    input_steplen = len(tr_input)
    output_steplen = len(tr_output)
    print(tr_input)
    print(tr_output)

    step_input = [dic_input_vocab["<start>"]] + [dic_input_vocab[word] for word in tr_input] + [dic_input_vocab["<end>"]]
    step_output =  [dic_output_vocab["<start>"]] + [dic_output_vocab[word] for word in tr_output] + [dic_output_vocab["<end>"]]

    train_input_tokens.append(step_input)
    train_output_tokens.append(step_output)
 
    input_max_len = input_max_len if input_max_len > input_steplen else input_steplen
    output_max_len = output_max_len if output_max_len > output_steplen else output_steplen


print('input_vocab size :', len(input_vocab))
print('output_vocab size :', len(output_vocab))

max_len = input_max_len if input_max_len > output_max_len else output_max_len

train_input_tokens = pad_sequences(train_input_tokens, max_len, padding = 'post')
train_output_tokens = pad_sequences(train_output_tokens, max_len, padding = 'post')

val_input_tokens = pad_sequences(val_input_tokens, max_len, padding = 'post')
val_output_tokens = pad_sequences(val_output_tokens, max_len, padding = 'post')

test_input_tokens = pad_sequences(test_input_tokens, max_len, padding = 'post')
test_output_tokens = pad_sequences(test_output_tokens, max_len, padding = 'post')

print(train_input_tokens[0])

with open("train_input_data.pickle", "wb") as fw:
	pickle.dump(train_input_tokens, fw)

with open("train_output_data.pickle", 'wb') as fw:
	pickle.dump(train_output_tokens, fw)

with open('val_input_tokens.pickle', 'wb') as fw:
	pickle.dump(val_input_tokens, fw)

with open('val_output_tokens.pickle', 'wb') as fw:
	pickle.dump(val_output_tokens, fw)

with open('test_input_tokens.pickle', 'wb') as fw:
	pickle.dump(test_input_tokens, fw)

with open('test_output_tokens.pickle', 'wb') as fw:
	pickle.dump(test_output_tokens, fw)

with open('input_vocab.pickle', 'wb') as fw:
	pickle.dump(input_vocab, fw)

with open('output_vocab.pickle', 'wb') as fw:
	pickle.dump(output_vocab, fw)
