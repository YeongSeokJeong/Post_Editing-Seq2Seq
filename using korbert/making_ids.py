#-*- coding:utf-8 -*-
import pickle as pkl
from tqdm import tqdm
import collections
import tensorflow as tf
from tokenization_morp import *
import numpy as np
import operator
from tensorflow.keras.preprocessing.sequence import pad_sequences
file_path = './data/'

with open(file_path + "train_input_tag.pkl", 'rb') as fr:
    train_input_morph = pkl.load(fr)
    
with open(file_path + "train_output_tag.pkl", 'rb') as fr:
    train_output_morph = pkl.load(fr)
    
with open(file_path + "val_input_tag.pkl", 'rb') as fr:
    val_input_morph = pkl.load(fr)
    
with open(file_path + "val_output_tag.pkl", 'rb') as fr:
    val_output_morph = pkl.load(fr)
    
with open(file_path + "test_input_tag.pkl", 'rb') as fr:
    test_input_morph = pkl.load(fr)
    
with open(file_path + "test_output_tag.pkl", 'rb') as fr:
    test_output_morph = pkl.load(fr)

tokenizer = FullTokenizer(vocab_file =  'vocab.korean_morp.list')

def sentence_tokenization(text, tokenizer):
	text = ['[CLS]'] + tokenizer.tokenize(text) + ['[SEP]']
	return tokenizer.convert_tokens_to_ids(text)

def corpus_tokenization(corpus, tokenizer):
	tokenized_corpus = []
	for sentence in corpus:
		tokenized_sent = sentence_tokenization(sentence, tokenizer)
		tokenized_corpus.append(tokenized_sent)
	return tokenized_corpus

def search_unk_count(tokens, unk_set):
    cnt = 0
    for i, sen in enumerate(tokens):
        for word in sen:
            if word == '[UNK]':
                unk_set.append(i)
                cnt += 1
                break
    return cnt, unk_set

train_input_token = corpus_tokenization(train_input_morph, tokenizer)
train_output_token = corpus_tokenization(train_output_morph, tokenizer)
val_input_token = corpus_tokenization(val_input_morph, tokenizer)
val_output_token = corpus_tokenization(val_output_morph, tokenizer)
test_input_token = corpus_tokenization(test_input_morph, tokenizer)
test_output_token = corpus_tokenization(test_output_morph, tokenizer)

train_unk_set = []
_, train_unk_set = search_unk_count(train_input_token, train_unk_set)
_, train_unk_set = search_unk_count(train_output_token, train_unk_set)
train_unk_set = set(train_unk_set)

val_unk_set = []
_, val_unk_set = search_unk_count(val_input_token, val_unk_set)
_, val_unk_set = search_unk_count(val_output_token, val_unk_set)
val_unk_set = set(val_unk_set)

test_unk_set = []
_, test_unk_set = search_unk_count(test_input_token, test_unk_set)
_, test_unk_set = search_unk_count(test_output_token, test_unk_set)
test_unk_set = set(test_unk_set)

def eliminate_sentence(corpus, unk_set):
	corpus = np.array(corpus)
	knw_set = [i for i in range(len(corpus)) if i not in unk_set]
	corpus = corpus[knw_set]
	return list(corpus)

train_input_token = eliminate_sentence(train_input_token, train_unk_set)
train_output_token = eliminate_sentence(train_output_token, train_unk_set)

val_input_token =  eliminate_sentence(val_input_token, val_unk_set)
val_output_token = eliminate_sentence(val_output_token, val_unk_set)

test_input_token = eliminate_sentence(test_input_token, test_unk_set)
test_output_token = eliminate_sentence(test_output_token, test_unk_set)

def cnt_max_len(corpus):
	return max([len(sentence) for sentence in corpus])

max_len = max([cnt_max_len(train_input_token), 
			  cnt_max_len(train_output_token), 
			  cnt_max_len(val_input_token), 
			  cnt_max_len(val_output_token), 
			  cnt_max_len(test_input_token), 
			  cnt_max_len(test_output_token)])

train_input_token = pad_sequences(train_input_token, max_len, padding = 'post')
train_output_token = pad_sequences(train_input_token, max_len, padding = 'post')
val_input_token = pad_sequences(train_input_token, max_len, padding = 'post')
val_output_token = pad_sequences(train_input_token, max_len, padding = 'post')
test_input_token = pad_sequences(train_input_token, max_len, padding = 'post')
test_output_token = pad_sequences(train_input_token, max_len, padding = 'post')

output_dir = './tokenized_data/'

with open(output_dir + "train_input_data.pickle", "wb") as fw:
	pkl.dump(train_input_token, fw)

with open(output_dir + "train_output_data.pickle", 'wb') as fw:
	pkl.dump(train_output_token, fw)

with open(output_dir + 'val_input_data.pickle', 'wb') as fw:
	pkl.dump(val_input_token, fw)

with open(output_dir + 'val_output_data.pickle', 'wb') as fw:
	pkl.dump(val_output_token, fw)

with open(output_dir + 'test_input_data.pickle', 'wb') as fw:
	pkl.dump(test_input_token, fw)

with open(output_dir + 'test_output_data.pickle', 'wb') as fw:
	pkl.dump(test_output_token, fw)