import tensorflow as tf

class CNNClassifier:

	def __init__(self,vector_size, img_classes):
		self.vector_size = vector_size
		self.img_classes = img_classes


	def define_model_net(self,img_features):
		# Convolutional Layer #1
		conv1 = tf.layers.conv2d(
			inputs=img_features,
			filters=32,
			kernel_size=[5, 5],
			padding="same",
			activation=tf.nn.relu)
		# Pooling Layer #1
		pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=1)

		# Convolutional Layer #2 and Pooling Layer #2
		conv2 = tf.layers.conv2d(
			inputs=pool1,
			filters=64,
			kernel_size=[5, 5],
			padding="same",
			activation=tf.nn.relu)
		pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

		# Dense Layer
		pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
		dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
		dropout = tf.layers.dropout(
			inputs=dense, rate=0.4, training=True)

		# Logits Layer
		self.logits = tf.layers.dense(inputs=dropout, units=self.img_classes)


	def __train_model_fn(self,image_features,image_labels,mode):
		img_features = tf.reshape(image_features,[-1,self.vector_size,self.vector_size,1])

		self.define_model_net(img_features)

		# Calculate Loss (for both TRAIN and EVAL modes)
		self.loss = tf.losses.softmax_cross_entropy(labels=image_labels, logits=self.logits)

		# Configure the Training Op (for TRAIN mode)
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(
			loss=self.loss,
			global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN, loss=self.loss, train_op=train_op)


	def __eval_model_fn(self,image_features,image_labels,mode):
		# Add evaluation metrics (for EVAL mode)
		self.predictions = {
			# Generate predictions (for PREDICT and EVAL mode)
			"classes": tf.argmax(input=self.logits, axis=1),
			# Add `softmax_tensor` to the graph. It is used for PREDICT and by the
			# `logging_hook`.
			"probabilities": tf.nn.softmax(self.logits, name="softmax_tensor")
		}
		eval_metric_ops = {
			"accuracy": tf.metrics.accuracy(
				labels=self.img_classes, predictions=self.predictions["classes"])}
		return tf.estimator.EstimatorSpec(
			mode=tf.estimator.ModeKeys.EVAL, loss=self.loss, eval_metric_ops=eval_metric_ops)


	def train_model(self):
		return tf.estimator.Estimator(
			model_fn = lambda features, labels, mode: self.__train_model_fn(features, labels, mode))

	def evaluate_model(self):
		return tf.estimator.Estimator(
			model_fn=lambda features, labels, mode: self.__eval_model_fn(features, labels, mode))


	def predict_image(self):
		predictions = {
			# Generate predictions (for PREDICT and EVAL mode)
			"classes": tf.argmax(input=self.logits, axis=1),
			# Add `softmax_tensor` to the graph. It is used for PREDICT and by the
			# `logging_hook`.
			"probabilities": tf.nn.softmax(self.logits, name="softmax_tensor")
		}
		tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT, predictions=predictions)
