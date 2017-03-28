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

# for test
def sample_y(m, n, ind):
	y = np.zeros([m,n])
	for i in range(m):
		y[i,ind] = 1
	return y

def concat(z,y):
	return tf.concat([z,y],1)

class CGAN():
	def __init__(self, generator, discriminator, data):
		self.generator = generator
		self.discriminator = discriminator
		self.data = data

		# data
		self.z_dim = self.data.z_dim
		self.y_dim = self.data.y_dim # condition
		self.X_dim = self.data.X_dim

		self.X = tf.placeholder(tf.float32, shape=[None, self.X_dim])
		self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
		self.y = tf.placeholder(tf.float32, shape=[None, self.y_dim])

		# nets
		self.G_sample = self.generator(concat(self.z, self.y))

		self.D_real, _ = self.discriminator(concat(self.X, self.y))
		self.D_fake, _ = self.discriminator(concat(self.G_sample, self.y), reuse = True)
		
		# loss
		self.D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real, labels=tf.ones_like(self.D_real))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.zeros_like(self.D_fake)))
		self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.ones_like(self.D_fake)))

		# solver
		self.D_solver = tf.train.AdamOptimizer().minimize(self.D_loss, var_list=self.discriminator.vars)
		self.G_solver = tf.train.AdamOptimizer().minimize(self.G_loss, var_list=self.generator.vars)
	
		for var in self.discriminator.vars:
			print var.name
			
		self.saver = tf.train.Saver()
		gpu_options = tf.GPUOptions(allow_growth=True)
		self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

	def train(self, sample_dir, ckpt_dir='ckpt', training_epoches = 1000000, batch_size = 64):
		fig_count = 0
		self.sess.run(tf.global_variables_initializer())
		
		for epoch in range(training_epoches):
			# update D
			X_b,y_b = self.data(batch_size)
			self.sess.run(
				self.D_solver,
				feed_dict={self.X: X_b, self.y: y_b, self.z: sample_z(batch_size, self.z_dim)}
				)
			# update G
			k = 1
			for _ in range(k):
				self.sess.run(
					self.G_solver,
					feed_dict={self.y:y_b, self.z: sample_z(batch_size, self.z_dim)}
				)
			
			# save img, model. print loss
			if epoch % 100 == 0 or epoch < 100:
				D_loss_curr = self.sess.run(
						self.D_loss,
            			feed_dict={self.X: X_b, self.y: y_b, self.z: sample_z(batch_size, self.z_dim)})
				G_loss_curr = self.sess.run(
						self.G_loss,
						feed_dict={self.y: y_b, self.z: sample_z(batch_size, self.z_dim)})
				print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'.format(epoch, D_loss_curr, G_loss_curr))

				if epoch % 1000 == 0:
					y_s = sample_y(16, self.y_dim, fig_count%10)
					samples = self.sess.run(self.G_sample, feed_dict={self.y: y_s, self.z: sample_z(16, self.z_dim)})

					fig = self.data.data2fig(samples)
					plt.savefig('{}/{}_{}.png'.format(sample_dir, str(fig_count).zfill(3), str(fig_count%10)), bbox_inches='tight')
					fig_count += 1
					plt.close(fig)

				#if epoch % 2000 == 0:
				#	self.saver.save(self.sess, os.path.join(ckpt_dir, "cgan.ckpt"))


if __name__ == '__main__':

	os.environ['CUDA_VISIBLE_DEVICES'] = '1'

	# save generated images
	sample_dir = 'Samples/mnist_cgan_mlp'
	if not os.path.exists(sample_dir):
		os.makedirs(sample_dir)

	# param
	generator = G_mlp_mnist()
	discriminator = D_mlp_mnist()

	data = mnist('mlp')

	# run
	cgan = CGAN(generator, discriminator, data)
	cgan.train(sample_dir)
