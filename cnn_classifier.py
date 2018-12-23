import tensorflow as tf

class CNNClassifier:

	def __init__(self,vector_size, img_classes):
		self.vector_size = vector_size
		self.img_classes = img_classes


	def define_model_net(self,img_features):
		# Convolutional Layer #1
		img_features = tf.cast(img_features, tf.float32)
		conv1 = tf.layers.conv2d(
			inputs=img_features,
			filters=5,
			kernel_size=[3, 3],
			padding="same",
			activation=tf.nn.relu)
		# Pooling Layer #1
		pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=1)

		# Convolutional Layer #2 and Pooling Layer #2
		conv2 = tf.layers.conv2d(
			inputs=pool1,
			filters=10,
			kernel_size=[3, 3],
			padding="same",
			activation=tf.nn.relu)
		pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

		print("train shape....")
		print(pool2.shape)

		# Dense Layer
		pool2_flat = tf.reshape(pool2, [-1, 99 * 99 * 10])
		dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
		dropout = tf.layers.dropout(
			inputs=dense, rate=0.4, training=True)

		# Logits Layer
		return tf.layers.dense(inputs=dropout, units=self.img_classes)



	def __model_fn(self,features, labels, mode, params):
		image_features = features["images"]
		img_features = tf.reshape(image_features, [-1, self.vector_size, self.vector_size, 1])
		logits = self.define_model_net(img_features)

		if mode == tf.estimator.ModeKeys.TRAIN:
			return self.__train_model_fn(labels, mode, params, logits)
		elif mode == tf.estimator.ModeKeys.EVAL:
			return self.__eval_model_fn(logits)
		else:
			return self.__predict_model_fn(logits)

	def __train_model_fn(self,image_labels,mode,params,logits):
		print(mode)
		print("training....")
		loss = tf.losses.softmax_cross_entropy(onehot_labels=image_labels, logits=logits)
		# Configure the Training Op (for TRAIN mode)
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(
				loss=loss,
				global_step=tf.train.get_or_create_global_step())
		return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=train_op)


	def __eval_model_fn(self,image_labels,logits):
		loss = tf.losses.softmax_cross_entropy(onehot_labels=image_labels, logits=logits)
		# Add evaluation metrics (for EVAL mode)
		self.predictions = {
			# Generate predictions (for PREDICT and EVAL mode)
			"classes": tf.argmax(input=logits, axis=1),
			# Add `softmax_tensor` to the graph. It is used for PREDICT and by the
			# `logging_hook`.
			"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
		}
		eval_metric_ops = {
				"accuracy": tf.metrics.accuracy(
					labels=self.img_classes, predictions=self.predictions["classes"])}
		return tf.estimator.EstimatorSpec(
				mode=tf.estimator.ModeKeys.EVAL, loss=loss, eval_metric_ops=eval_metric_ops)

	def __predict_model_fn(self,logits):
		print("PRED....")
		print(logits)
		predictions = {
				"classes": tf.argmax(logits, axis=1),
				"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
			}
		return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT,
										  predictions=predictions,
										  export_outputs={
											  'classify': tf.estimator.export.PredictOutput(predictions)
										  })


	def get_classifier_model(self):
		print("get the model...")
		return tf.estimator.Estimator(
			model_fn = lambda features, labels, mode, params: self.__model_fn(features, labels, mode, params), model_dir="/tmp/cnn_data")