import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os,sys

sys.path.append('utils')
from nets import *
from datas import *

def sample_z(m, n):
	return np.random.uniform(-1., 1., size=[m, n])

class DCGAN():
	def __init__(self, generator, discriminator, data):
		self.generator = generator
		self.discriminator = discriminator
		self.data = data

		# data
		self.z_dim = self.data.z_dim
		self.size = self.data.size
		self.channel = self.data.channel

		self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
		self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])

		# nets
		self.G_sample = self.generator(self.z)

		self.D_real, _ = self.discriminator(self.X)
		self.D_fake, _ = self.discriminator(self.G_sample, reuse = True)
		
		# loss
		self.D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real, labels=tf.ones_like(self.D_real))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.zeros_like(self.D_fake)))
		self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.ones_like(self.D_fake)))

		# solver
		self.D_solver = tf.train.AdamOptimizer(learning_rate=2e-4).minimize(self.D_loss, var_list=self.discriminator.vars)
		self.G_solver = tf.train.AdamOptimizer(learning_rate=2e-4).minimize(self.G_loss, var_list=self.generator.vars)
		
		self.saver = tf.train.Saver()
		gpu_options = tf.GPUOptions(allow_growth=True)
		self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

	def train(self, sample_dir, ckpt_dir='ckpt', training_epoches = 1000000, batch_size = 32):
		fig_count = 0
		self.sess.run(tf.global_variables_initializer())
		
		for epoch in range(training_epoches):
			# update D
			X_b = self.data(batch_size)
			self.sess.run(
				self.D_solver,
				feed_dict={self.X: X_b, self.z: sample_z(batch_size, self.z_dim)}
				)
			# update G
			k = 1
			for _ in range(k):
				self.sess.run(
					self.G_solver,
					feed_dict={self.z: sample_z(batch_size, self.z_dim)}
				)

			# save img, model. print loss
			if epoch % 100 == 0 or epoch < 100:
				D_loss_curr = self.sess.run(
						self.D_loss,
            			feed_dict={self.X: X_b, self.z: sample_z(batch_size, self.z_dim)})
				G_loss_curr = self.sess.run(
						self.G_loss,
						feed_dict={self.z: sample_z(batch_size, self.z_dim)})
				print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'.format(epoch, D_loss_curr, G_loss_curr))

				if epoch % 1000 == 0:
					samples = self.sess.run(self.G_sample, feed_dict={self.z: sample_z(16, self.z_dim)})

					fig = self.data.data2fig(samples)
					plt.savefig('{}/{}.png'.format(sample_dir, str(fig_count).zfill(3)), bbox_inches='tight')
					fig_count += 1
					plt.close(fig)

				#if epoch % 2000 == 0:
					#self.saver.save(self.sess, os.path.join(ckpt_dir, "dcgan.ckpt"))


if __name__ == '__main__':

	# constraint GPU
	os.environ['CUDA_VISIBLE_DEVICES'] = '3'

	# save generated images
	sample_dir = 'Samples/celebA_dcgan'
	if not os.path.exists(sample_dir):
		os.makedirs(sample_dir)

	# param
	generator = G_conv()
	discriminator = D_conv()

	data = celebA()

	# run
	dcgan = DCGAN(generator, discriminator, data)
	dcgan.train(sample_dir)

