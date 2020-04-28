import tensorflow as tf
class Train(object):
	def __init__(self, epochs, encoder, decoder, inp_vocab, targ_vocab, batch_size, per_replica_batch_size):

		self.epochs = epochs
		self.encoder = encoder
		self.decoder = decoder
		self.inp_lang = inp_lang
		self.targ_lang = targ_lang

		self.batch_size = batch_size
		self.per_replica_batch_size = per_rablica_batch_size

		self.optimizer = tf.keras.optimizer.Adam()
		self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
			from_logits = True, reduction = tf.keras.losses.Reduction.None)

		self.train_loss_metric = tf.keras.metrics.Mean()

	def loss_function(self, real, pred):
		mask = tf.math.logical_not(tf.math.equal(real, 0))
		loss_ = self.loss_object(real, pred)

		mask = tf.cast(mask, dtype=loss_.dtype)
		loss_ *= mask

		return tf.reduce_sum(loss_) * 1. / self.batch_size

	def train_step(self, inputs):
		loss = 0
		enc_hidden = encoder.initialize_hidden_state()

		inp, targ = inputs

		with tf.Gradienttape() as tape:
			enc_output, enc_hidden = self.encoder(inp, enc_hidden)
			dec_hidden = enc_hidden
			dec_input = tf.expand_dims([self.targ_lang["[CLS]"]] * self.batch_size, 1)

			for t in range(1, targ.shape[1]):
				predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
				loss += self.loss_function(targ[:, t], predictions)
				dec_input = tf.expand_dims(targ[:, t], 1)

			batch_loss = (loss / int(targ.shape[1]))
			variables = (self.encoder.trainable_variables + self.decoder.trainable_variables)
			gradients = tape.gradient(loss, variables)

			self.optimizer.apply_gradients(zip(gradients, variables))

			self.train_loss_metric(batch_loss)

	def train_loop(self, train_ds, val_ds):
		self.train_step = tf.function(self.train_step)
		self.test_step = tf.function(self.test_step)
		output_text = 'Epoch P{}, Train_loss : {}'

		for epoch in range(self.epochs):
			self.train_loss_metric.reset_states()

			for inp, targ in train_ds:
				self.train_step((inp, targ))

			print(output_text.format(self.train_loss_metric.result().numpy()))

		return self.train_loss_metric.result().numpy()







