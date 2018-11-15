import tensorflow as tf

class CNNClassifier:

	def __init__(self,img_size,num_channels,img_classes):
		self.img_size = img_size
		self.num_channels = num_channels
		self.img_classes = img_classes

	def create_features(self):
		self.image_features = tf.placeholder(tf.float32, shape = [-1,self.img_size,self.img_size], name="image features")
		self.image_labels = tf.placeholder(tf.float32, shape = [-1,self.img_classes], name="image classes")


	def train_classifier(self):
		# Convolutional Layer #1
		conv1 = tf.layers.conv2d(
			inputs=self.image_features,
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
			inputs=dense, rate=0.4, training=tf.estimator.ModeKeys.TRAIN)

		# Logits Layer
		self.logits = tf.layers.dense(inputs=dropout, units=self.img_classes)

		# Calculate Loss (for both TRAIN and EVAL modes)
		self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.image_labels, logits=self.logits)

		# Configure the Training Op (for TRAIN mode)
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(
			loss=self.loss,
			global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN, loss=self.loss, train_op=train_op)


	def evaluate_model(self):
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


	def predict_image(self):
		predictions = {
			# Generate predictions (for PREDICT and EVAL mode)
			"classes": tf.argmax(input=self.logits, axis=1),
			# Add `softmax_tensor` to the graph. It is used for PREDICT and by the
			# `logging_hook`.
			"probabilities": tf.nn.softmax(self.logits, name="softmax_tensor")
		}
		tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT, predictions=predictions)
