from khaiii import KhaiiiApi
import pickle as pkl
from addition_data import * 
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

path_dir = './data/shuffle_char_output/'
out_path_dir = './data/khaiii_output/'
api = KhaiiiApi()

with open(path_dir + 'train_input.pkl', 'rb') as fr:
	train_input = pkl.load(fr)

with open(path_dir + 'train_output.pkl', 'rb') as fr:
	train_output = pkl.load(fr)

with open(path_dir + 'val_input.pkl', 'rb') as fr:
	val_input = pkl.load(fr)

with open(path_dir + 'val_output.pkl', 'rb') as fr:
	val_output = pkl.load(fr)

with open(path_dir + 'test_input.pkl', 'rb') as fr:
	test_input = pkl.load(fr)

with open(path_dir + 'test_output.pkl', 'rb') as fr:
	test_output = pkl.load(fr)

train_input_morph = []
train_output_morph = []

val_input_morph = []
val_output_morph = []

test_input_morph = []
test_output_morph = []

for sentence in tqdm(train_input):
	li_sen = []
	for word in api.analyze(sentence):
		for morph in word.morphs:
			li_sen.append(str(morph))
	train_input_morph.append(li_sen)

for sentence in tqdm(train_output):
	li_sen = []
	for word in api.analyze(sentence):
		for morph in word.morphs:
			li_sen.append(str(morph))
	train_output_morph.append(li_sen)

for sentence in tqdm(val_input):
	li_sen = []
	for word in api.analyze(sentence):
		for morph in word.morphs:
			li_sen.append(str(morph))
	val_input_morph.append(li_sen)

for sentence in tqdm(val_output):
	li_sen = []
	for word in api.analyze(sentence):
		for morph in word.morphs:
			li_sen.append(str(morph))
	val_output_morph.append(li_sen)

for sentence in tqdm(test_input):
	li_sen = []
	for word in api.analyze(sentence):
		for morph in word.morphs:
			li_sen.append(str(morph))
	test_input_morph.append(li_sen)

for sentence in tqdm(test_output):
	li_sen = []
	for word in api.analyze(sentence):
		for morph in word.morphs:
			li_sen.append(str(morph))
	test_output_morph.append(li_sen)

with open("./data/train_input_tag.pkl", "wb") as fw:
	pkl.dump(train_input_morph, fw)

with open("./data/train_output_tag.pkl", 'wb') as fw:
	pkl.dump(train_output_morph, fw)

with open('./data/val_input_tag.pkl', 'wb') as fw:
	pkl.dump(val_input_morph, fw)

with open('./data/val_output_tag.pkl', 'wb') as fw:
	pkl.dump(val_output_morph, fw)

with open('./data/test_input_tag.pkl', 'wb') as fw:
	pkl.dump(test_input_morph, fw)

with open('./data/test_output_tag.pkl', 'wb') as fw:
	pkl.dump(test_output_morph, fw)