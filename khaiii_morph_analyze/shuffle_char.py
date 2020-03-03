import pandas as pd
import numpy as np
from addition_data import make_char_data
import pickle as pkl
from sklearn.model_selection import train_test_split

data = pd.read_csv("data_new7.csv",encoding = 'cp949')

input_data = data.iloc[:, 0].to_list()
output_data = data.iloc[:, 1].to_list()

train_input, test_input, train_output, test_output = train_test_split(input_data,
                                                                      output_data,
                                                                      test_size = 0.2)

train_input, val_input, train_output, val_output = train_test_split(train_input,
                                                                    train_output,
                                                                    test_size = 0.125)

train_input, train_output = make_char_data(train_input, train_output, int(len(train_input) / 2))

with open("./data/shuffle_char_output/train_input.pkl", "wb") as fw:
	pkl.dump(train_input, fw)

with open("./data/shuffle_char_output/train_output.pkl", 'wb') as fw:
	pkl.dump(train_output, fw)

with open('./data/shuffle_char_output/val_input.pkl', 'wb') as fw:
	pkl.dump(val_input, fw)

with open('./data/shuffle_char_output/val_output.pkl', 'wb') as fw:
	pkl.dump(val_output, fw)

with open('./data/shuffle_char_output/test_input.pkl', 'wb') as fw:
	pkl.dump(test_input, fw)

with open('./data/shuffle_char_output/test_output.pkl', 'wb') as fw:
	pkl.dump(test_output, fw)