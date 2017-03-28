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

class WGAN():
	def __init__(self, generator, discriminator, data):
		self.generator = generator
		self.discriminator = discriminator
		self.data = data

		self.z_dim = self.data.z_dim
		self.X_dim = self.data.X_dim

		self.X = tf.placeholder(tf.float32, shape=[None, self.X_dim])
		self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])

		# nets
		self.G_sample = self.generator(self.z)

		self.D_real, _ = self.discriminator(self.X)
		self.D_fake, _ = self.discriminator(self.G_sample, reuse = True)

		# loss
		self.D_loss = - tf.reduce_mean(self.D_real) + tf.reduce_mean(self.D_fake)
		self.G_loss = - tf.reduce_mean(self.D_fake)

		self.D_solver = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(self.D_loss, var_list=self.discriminator.vars)
		self.G_solver = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(self.G_loss, var_list=self.generator.vars)
		
		# clip
		self.clip_D = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.discriminator.vars]
		
		gpu_options = tf.GPUOptions(allow_growth=True)
		self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

	def train(self, sample_folder, training_epoches = 1000000, batch_size = 64):
		i = 0
		self.sess.run(tf.global_variables_initializer())
		
		for epoch in range(training_epoches):
			# update D
			n_d = 100 if epoch < 25 or (epoch+1) % 500 == 0 else 5
			for _ in range(n_d):
				X_b, _ = self.data(batch_size)
				self.sess.run(self.clip_D)
				self.sess.run(
						self.D_solver,
            			feed_dict={self.X: X_b, self.z: sample_z(batch_size, self.z_dim)}
						)
			# update G
			self.sess.run(
				self.G_solver,
				feed_dict={self.z: sample_z(batch_size, self.z_dim)}
			)

			# print loss. save images.
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
					plt.savefig('{}/{}.png'.format(sample_folder, str(i).zfill(3)), bbox_inches='tight')
					i += 1
					plt.close(fig)


if __name__ == '__main__':

	os.environ['CUDA_VISIBLE_DEVICES'] = '1'
	
	sample_folder = 'Samples/mnist_wgan_mlp'
	if not os.path.exists(sample_folder):
		os.makedirs(sample_folder)

	# param
	generator = G_mlp_mnist()
	discriminator = D_mlp_mnist()

	data = mnist('mlp')

	# run
	wgan = WGAN(generator, discriminator, data)
	wgan.train(sample_folder)

