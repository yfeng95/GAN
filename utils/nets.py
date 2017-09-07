import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl

def leaky_relu(x, alpha=0.2):
	return tf.maximum(tf.minimum(0.0, alpha * x), x
)

def lrelu(x, leak=0.2, name="lrelu"):
	with tf.variable_scope(name):
		f1 = 0.5 * (1 + leak)
		f2 = 0.5 * (1 - leak)
		return f1 * x + f2 * abs(x)

def xavier_init(size):
	in_dim = size[0]
	xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
	return tf.random_normal(shape=size, stddev=xavier_stddev)

###############################################  mlp #############################################
class G_mlp(object):
	def __init__(self):
		self.name = 'G_mlp'

	def __call__(self, z):
		with tf.variable_scope(self.name) as scope:
			g = tcl.fully_connected(z, 4 * 4 * 512, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
			g = tcl.fully_connected(g, 64, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
			g = tcl.fully_connected(g, 64, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
			g = tcl.fully_connected(g, 64*64*3, activation_fn=tf.nn.tanh, normalizer_fn=tcl.batch_norm)
			g = tf.reshape(g, tf.stack([tf.shape(z)[0], 64, 64, 3]))
			return g
	@property
	def vars(self):
		return [var for var in tf.global_variables() if self.name in var.name]

class D_mlp(object):
	def __init__(self):
		self.name = "D_mlp"

	def __call__(self, x, reuse=True):
		with tf.variable_scope(self.name) as vs:
			if reuse:
				vs.reuse_variables()
			d = tcl.fully_connected(tf.flatten(x), 64, activation_fn=tf.nn.relu,normalizer_fn=tcl.batch_norm)
			d = tcl.fully_connected(d, 64,activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
			d = tcl.fully_connected(d, 64,activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
			logit = tcl.fully_connected(d, 1, activation_fn=None)

		return logit

	@property
	def vars(self):
		return [var for var in tf.global_variables() if self.name in var.name]

#-------------------------------- MNIST for test ------
class G_mlp_mnist(object):
	def __init__(self):
		self.name = "G_mlp_mnist"
		self.X_dim = 784

	def __call__(self, z):
		with tf.variable_scope(self.name) as vs:
			g = tcl.fully_connected(z, 128, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(0, 0.02))
			g = tcl.fully_connected(g, self.X_dim, activation_fn=tf.nn.sigmoid, weights_initializer=tf.random_normal_initializer(0, 0.02))
		return g

	@property
	def vars(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class D_mlp_mnist():
	def __init__(self):
		self.name = "D_mlp_mnist"

	def __call__(self, x, reuse=False):
		with tf.variable_scope(self.name) as scope:
			if reuse:
				scope.reuse_variables()
			shared = tcl.fully_connected(x, 128, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(0, 0.02))
			d = tcl.fully_connected(shared, 1, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02))
			
			q = tcl.fully_connected(shared, 10, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02)) # 10 classes
			
		return d, q

	@property
	def vars(self):		
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class Q_mlp_mnist():
	def __init__(self):
		self.name = "Q_mlp_mnist"

	def __call__(self, x, reuse=False):
		with tf.variable_scope(self.name) as scope:
			if reuse:
				scope.reuse_variables()
			shared = tcl.fully_connected(x, 128, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(0, 0.02))
			q = tcl.fully_connected(shared, 10, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02)) # 10 classes
		return q

	@property
	def vars(self):		
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


###############################################  conv #############################################
class G_conv(object):
	def __init__(self):
		self.name = 'G_conv'
		self.size = 64/16
		self.channel = 3

	def __call__(self, z):
		with tf.variable_scope(self.name) as scope:
			g = tcl.fully_connected(z, self.size * self.size * 1024, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
			g = tf.reshape(g, (-1, self.size, self.size, 1024))  # size
			g = tcl.conv2d_transpose(g, 512, 3, stride=2, # size*2
									activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
			g = tcl.conv2d_transpose(g, 256, 3, stride=2, # size*4
									activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
			g = tcl.conv2d_transpose(g, 128, 3, stride=2, # size*8
									activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
			
			g = tcl.conv2d_transpose(g, self.channel, 3, stride=2, # size*16
										activation_fn=tf.nn.sigmoid, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
			return g
	@property
	def vars(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class D_conv(object):
	def __init__(self):
		self.name = 'D_conv'

	def __call__(self, x, reuse=False):
		with tf.variable_scope(self.name) as scope:
			if reuse:
				scope.reuse_variables()
			size = 64
			shared = tcl.conv2d(x, num_outputs=size, kernel_size=4, # bzx64x64x3 -> bzx32x32x64
						stride=2, activation_fn=lrelu)
			shared = tcl.conv2d(shared, num_outputs=size * 2, kernel_size=4, # 16x16x128
						stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
			shared = tcl.conv2d(shared, num_outputs=size * 4, kernel_size=4, # 8x8x256
						stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
			shared = tcl.conv2d(shared, num_outputs=size * 8, kernel_size=4, # 4x4x512
						stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)

			shared = tcl.flatten(shared)
	
			d = tcl.fully_connected(shared, 1, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02))
			q = tcl.fully_connected(shared, 128, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
			q = tcl.fully_connected(q, 2, activation_fn=None) # 10 classes
			return d, q
			
	@property
	def vars(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class C_conv(object):
	def __init__(self):
		self.name = 'C_conv'

	def __call__(self, x, reuse=False):
		with tf.variable_scope(self.name) as scope:
			if reuse:
				scope.reuse_variables()
			size = 64
			shared = tcl.conv2d(x, num_outputs=size, kernel_size=4, # bzx64x64x3 -> bzx32x32x64
						stride=2, activation_fn=lrelu)
			shared = tcl.conv2d(shared, num_outputs=size * 2, kernel_size=4, # 16x16x128
						stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
			shared = tcl.conv2d(shared, num_outputs=size * 4, kernel_size=4, # 8x8x256
						stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
			#d = tcl.conv2d(d, num_outputs=size * 8, kernel_size=3, # 4x4x512
			#			stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)

			shared = tcl.fully_connected(tcl.flatten( # reshape, 1
						shared), 1024, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
			
			q = tcl.fully_connected(tcl.flatten(shared), 128, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
			q = tcl.fully_connected(q, 10, activation_fn=None) # 10 classes
		
			return q
	@property
	def vars(self):
		return [var for var in tf.global_variables() if self.name in var.name]

class V_conv(object):
	def __init__(self):
		self.name = 'V_conv'

	def __call__(self, x, reuse=False):
		with tf.variable_scope(self.name) as scope:
			if reuse:
				scope.reuse_variables()
			size = 64
			shared = tcl.conv2d(x, num_outputs=size, kernel_size=4, # bzx64x64x3 -> bzx32x32x64
						stride=2, activation_fn=tf.nn.relu)
			shared = tcl.conv2d(shared, num_outputs=size * 2, kernel_size=4, # 16x16x128
						stride=2, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
			shared = tcl.conv2d(shared, num_outputs=size * 4, kernel_size=4, # 8x8x256
						stride=2, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
			shared = tcl.conv2d(shared, num_outputs=size * 8, kernel_size=3, # 4x4x512
						stride=2, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)

			shared = tcl.fully_connected(tcl.flatten( # reshape, 1
						shared), 1024, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
			
			v = tcl.fully_connected(tcl.flatten(shared), 128)
			return v
	@property
	def vars(self):
		return [var for var in tf.global_variables() if self.name in var.name]


# -------------------------------- MNIST for test
class G_conv_mnist(object):
	def __init__(self):
		self.name = 'G_conv_mnist'

	def __call__(self, z):
		with tf.variable_scope(self.name) as scope:
			#g = tcl.fully_connected(z, 1024, activation_fn = tf.nn.relu, normalizer_fn=tcl.batch_norm,
			#						weights_initializer=tf.random_normal_initializer(0, 0.02))
			g = tcl.fully_connected(z, 7*7*128, activation_fn = tf.nn.relu, normalizer_fn=tcl.batch_norm,
									weights_initializer=tf.random_normal_initializer(0, 0.02))
			g = tf.reshape(g, (-1, 7, 7, 128))  # 7x7
			g = tcl.conv2d_transpose(g, 64, 4, stride=2, # 14x14x64
									activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
			g = tcl.conv2d_transpose(g, 1, 4, stride=2, # 28x28x1
										activation_fn=tf.nn.sigmoid, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
			return g
	@property
	def vars(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
	
class D_conv_mnist(object):
	def __init__(self):
		self.name = 'D_conv_mnist'

	def __call__(self, x, reuse=False):
		with tf.variable_scope(self.name) as scope:
			if reuse:
				scope.reuse_variables()
			size = 64
			shared = tcl.conv2d(x, num_outputs=size, kernel_size=4, # bzx28x28x1 -> bzx14x14x64
						stride=2, activation_fn=lrelu)
			shared = tcl.conv2d(shared, num_outputs=size * 2, kernel_size=4, # 7x7x128
						stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
			shared = tcl.flatten(shared)
			
			d = tcl.fully_connected(shared, 1, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02))
			q = tcl.fully_connected(shared, 128, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
			q = tcl.fully_connected(q, 10, activation_fn=None) # 10 classes
			return d, q
	@property
	def vars(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class C_conv_mnist(object):
	def __init__(self):
		self.name = 'C_conv_mnist'

	def __call__(self, x, reuse=False):
		with tf.variable_scope(self.name) as scope:
			if reuse:
				scope.reuse_variables()
			size = 64
			shared = tcl.conv2d(x, num_outputs=size, kernel_size=5, # bzx28x28x1 -> bzx14x14x64
						stride=2, activation_fn=tf.nn.relu)
			shared = tcl.conv2d(shared, num_outputs=size * 2, kernel_size=5, # 7x7x128
						stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
			shared = tcl.fully_connected(tcl.flatten( # reshape, 1
						shared), 1024, activation_fn=tf.nn.relu)
			
			#c = tcl.fully_connected(shared, 128, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
			c = tcl.fully_connected(shared, 10, activation_fn=None) # 10 classes
			return c
	@property
	def vars(self):
		return [var for var in tf.global_variables() if self.name in var.name]

