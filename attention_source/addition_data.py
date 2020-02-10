from random import sample
from konlpy.tag import Kkma
import pandas as pd

def make_char_data(input_data, output_data, change_num):

	char = [ch for sen in output_data for ch in sen]
	voc_char = set(char)

	voc_char = [ch for ch in voc_char if not ((ord(ch) > 97 and ord(ch) < 122) or (ord(ch) > 65 and ord(ch) < 90))]

	del voc_char[voc_char.index(' ')]

	num_list = [i for i in range(len(output_data))]

	sample_list = sample(num_list, change_num)
	
	new_input = []
	new_output = []
	for i in range(len(sample_list)):
		sen = output_data[sample_list[i]]

		sen1 = changing_char(voc_char, sen, 1)
		sen3 = changing_char(voc_char, sen, 3)

		new_input.append(sen1)
		new_input.append(sen3)
		new_output.append(output_data[sample_list[i]])
		new_output.append(output_data[sample_list[i]])

	for i in range(len(new_input)):
		input_data.append(new_input[i])
		output_data.append(new_output[i])
	return input_data, output_data

def changing_char(vocab, sentence, option):
	new_sen = sentence
	num_sen = [i for i in range(len(sentence))]
	num_list = []
	while True:
	    idx = sample(num_sen, 1)[0]
	    if idx in num_list:
	        continue
	    if sentence[idx] == ' ':
	        continue
	    new_char = sample(vocab, 1)[0]
	    if sentence[idx] == new_char:
	        continue
	    new_sen = new_sen[:idx] + new_char + new_sen[idx + 1:]
	    num_list.append(idx)
	    if len(num_list) == option:
	        break
	return new_sen

def make_morph_data(input_data, output_data, output_vocab, change_num):
	output_vocab = list(output_vocab)
	num_list = [i for i in range(len(output_data))]

	sample_list = sample(num_list, change_num)

	new_input, new_output = [], []

	for i in range(len(sample_list)):
		sen = output_data[sample_list[i]].copy()

		sen1 = changing_morph(output_vocab, sen, 1)
		sen3 = changing_morph(output_vocab, sen, 3)
		new_input.append(sen1)
		new_input.append(sen3)
		new_output.append(output_data[sample_list[i]])
		new_output.append(output_data[sample_list[i]])
		break
		if i % 1000 == 0 : 
			print('{} complete!'.format(i))

	for i in range(len(new_input)):
		input_data.append(new_input[i])
		output_data.append(new_output[i])
	return input_data, output_data

def changing_morph(vocab, sentence, option):
	new_sen = sentence.copy()
	num_sen = [i for i in range(len(sentence))]
	num_list = []
	if len(new_sen) < option:
		return new_sen
	while True:
		idx = sample(num_sen, 1)[0]
		if idx in num_list:
			continue
		new_morph = sample(vocab, 1)[0]
		if new_sen[idx] == new_morph:
			continue
		new_sen[idx] = new_morph
		num_list.append(idx)
		if len(num_list) == option:
			break
	return new_sen
