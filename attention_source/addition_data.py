from random import sample
from konlpy.tag import Kkma
import pandas as pd

def make_char_data(input_data, output_data):

	char = [ch for sen in input_data for ch in sen]
	voc_char = list(set(char))

	del voc_char[voc_char.index(' ')]

	num_list = [i for i in range(len(input_data))]

	sample_list = sample(li, 10000)
	
	new_input = []
	new_output = []
	for i in range(len(sample_list)):
		sen = input_data[sample_list[i]]

		sen1 = changing_char(voc_char, sen1, 1)
		sen3 = changing_char(voc_char, sen1, 3)

		new_input.append(sen1)
		new_input.append(sen3)
		new_output.append(output_data[sample_list[i]])
		new_output.append(output_data[sample_list[i]])
		
	for i in range(len(new_input)):
		input_data.append(new_input[i])
		output_data.append(new_output[i])
	return new_input, new_output

def changing_char(vocab, sentence, option):
	num_sen = [i for i in range(len(sen1))]
	num_list = []
	while True:
		idx = sample(num_sen, 1)
		if idx in num_list:
			continue
		if sen1[idx] == ' ':
			continue
		new_char = vocab[sample(vocab, 1)]
		if sentence[idx] == new_char:
			continue
		sentence[idx] = new_char
		num_list.append(li)
		if len(num_list) == option:
			break
	return sentence