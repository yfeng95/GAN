import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys

sys.path.append('utils')
from nets import *
from datas import *

def sample_z(m, n):
	return np.random.uniform(-1., 1., size=[m, n])

def sample_c(m, n, ind=-1):
	c = np.zeros([m,n])
	for i in range(m):
		if ind<0:
			ind = np.random.randint(10)
		c[i,i%10] = 1
	return c

def concat(z,c):
	return tf.concat([z,c],1)

class InfoGAN():
	def __init__(self, generator, discriminator, data):
		self.generator = generator
		self.discriminator = discriminator
		self.data = data

		# data
		self.z_dim = self.data.z_dim
		self.c_dim = self.data.y_dim # condition
		self.size = self.data.size
		self.channel = self.data.channel

		self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
		self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
		self.c = tf.placeholder(tf.float32, shape=[None, self.c_dim])

		# nets
		# G
		self.G_sample = self.generator(concat(self.z, self.c))
		# D and Q
		self.D_real, _ = self.discriminator(self.X)
		self.D_fake, self.Q_fake = self.discriminator(self.G_sample, reuse = True)
		
		# loss
		self.D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real, labels=tf.ones_like(self.D_real))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.zeros_like(self.D_fake)))
		self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.ones_like(self.D_fake)))
		self.Q_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.Q_fake, labels=self.c))

		# solver
		self.D_solver = tf.train.AdamOptimizer().minimize(self.D_loss, var_list=self.discriminator.vars)
		self.G_solver = tf.train.AdamOptimizer().minimize(self.G_loss, var_list=self.generator.vars)
		self.Q_solver = tf.train.AdamOptimizer().minimize(self.Q_loss, var_list=self.generator.vars + self.discriminator.vars)
		
		self.saver = tf.train.Saver()
		gpu_options = tf.GPUOptions(allow_growth=True)
		self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

	def train(self, sample_dir, ckpt_dir='ckpt', training_epoches = 1000000, batch_size = 64):
		fig_count = 0
		self.sess.run(tf.global_variables_initializer())
		
		for epoch in range(training_epoches):
			X_b, _= self.data(batch_size)
			z_b = sample_z(batch_size, self.z_dim)
			c_b = sample_c(batch_size, self.c_dim)
			# update D
			self.sess.run(
				self.D_solver,
				feed_dict={self.X: X_b, self.z: z_b, self.c: c_b}
				)
			# update G
			for _ in range(1):
				self.sess.run(
					self.G_solver,
					feed_dict={self.z: z_b, self.c: c_b}
				)
			# update Q
			for _ in range(2):	
				self.sess.run(
					self.Q_solver,
					feed_dict={self.z: z_b, self.c: c_b}
				)
			# save img, model. print loss
			if epoch % 100 == 0 or epoch < 100:
				D_loss_curr = self.sess.run(
						self.D_loss,
            			feed_dict={self.X: X_b, self.z: z_b, self.c: c_b})
				G_loss_curr, Q_loss_curr = self.sess.run(
						[self.G_loss, self.Q_loss],
						feed_dict={self.z: z_b, self.c: c_b})
				print('Iter: {}; D loss: {:.4}; G_loss: {:.4}; Q_loss: {:.4}'.format(epoch, D_loss_curr, G_loss_curr, Q_loss_curr))

				if epoch % 1000 == 0:
					z_s = sample_z(16, self.z_dim)
					c_s = sample_c(16, self.c_dim, fig_count%10)
					samples = self.sess.run(self.G_sample, feed_dict={self.c: c_s, self.z: z_s})

					fig = self.data.data2fig(samples)
					plt.savefig('{}/{}_{}.png'.format(sample_dir, str(fig_count).zfill(3), str(fig_count%10)), bbox_inches='tight')
					fig_count += 1
					plt.close(fig)

				#if epoch % 2000 == 0:
				#	self.saver.save(self.sess, os.path.join(ckpt_dir, "infogan.ckpt"))


if __name__ == '__main__':

	os.environ['CUDA_VISIBLE_DEVICES'] = '3'

	# save generated images
	sample_dir = 'Samples/mnist_infogan_conv'
	if not os.path.exists(sample_dir):
		os.makedirs(sample_dir)

	# param
	generator = G_conv_mnist()
	discriminator = D_conv_mnist()
	data = mnist()

	# run
	infogan = InfoGAN(generator, discriminator, data)
	infogan.train(sample_dir)

