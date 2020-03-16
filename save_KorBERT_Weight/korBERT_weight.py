import tensorflow as tf
from modeling import *
from tokenization_morp import *

def save_weight(bert_config, is_training, input_ids, input_mask, 
	token_type_ids, use_one_hot_embedding, init_checkpoint):
	input_ids = tf.convert_to_tensor([input_ids])
	input_mask = tf.convert_to_tensor([input_mask])
	i_segment_ids = tf.convert_to_tensor([i_segment_ids])

	model = modeling.BertModel(config = bert_config,
						   is_training = False,
						   input_ids = input_ids,
						   input_mask = input_mask,
						   token_type_ids = i_segment_ids,
						   use_one_hot_embeddings = False)

	init_checkpoint = '/content/drive/My Drive/weight/model.ckpt'

	tvars = tf.trainable_variables()

	initialized_varable_names = {}

	(assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

	tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

	output_layer = model.get_embedding_table()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver({'word_embeddings' : output_layer})
		save_path = saver.save(sess, './embedding_weight.ckpt')

	return output_layer