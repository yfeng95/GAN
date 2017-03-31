import os,sys
from PIL import Image
import scipy.misc
from glob import glob
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tensorflow.examples.tutorials.mnist import input_data

prefix = './Datas/'
def get_img(img_path, crop_h, resize_h):
	img=scipy.misc.imread(img_path).astype(np.float)
	# crop resize
	crop_w = crop_h
	#resize_h = 64
	resize_w = resize_h
	h, w = img.shape[:2]
	j = int(round((h - crop_h)/2.))
	i = int(round((w - crop_w)/2.))
	cropped_image = scipy.misc.imresize(img[j:j+crop_h, i:i+crop_w],[resize_h, resize_w])

	return np.array(cropped_image)/255.0

class face3D():
	def __init__(self):
		datapath = '/ssd/fengyao/pose/pose/images'
		self.z_dim = 100
		self.c_dim = 2
		self.size = 64
		self.channel = 3
		self.data = glob(os.path.join(datapath, '*.jpg'))

		self.batch_count = 0

	def __call__(self,batch_size):
		batch_number = len(self.data)/batch_size
		if self.batch_count < batch_number-1:
			self.batch_count += 1
		else:
			self.batch_count = 0

		path_list = self.data[self.batch_count*batch_size:(self.batch_count+1)*batch_size]

		batch = [get_img(img_path, 256, self.size) for img_path in path_list]
		batch_imgs = np.array(batch).astype(np.float32)
		#fig = self.data2fig(batch_imgs[:16,:,:])
		#plt.savefig('out_face/{}.png'.format(str(self.batch_count).zfill(3)), bbox_inches='tight')
		#plt.close(fig)
		
		return batch_imgs

	def data2fig(self, samples):
		fig = plt.figure(figsize=(4, 4))
		gs = gridspec.GridSpec(4, 4)
		gs.update(wspace=0.05, hspace=0.05)

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample)
		return fig

class celebA():
	def __init__(self):
		datapath = prefix + 'celebA'
		self.z_dim = 100
		self.size = 64
		self.channel = 3
		self.data = glob(os.path.join(datapath, '*.jpg'))

		self.batch_count = 0

	def __call__(self,batch_size):
		batch_number = len(self.data)/batch_size
		if self.batch_count < batch_number-1:
			self.batch_count += 1
		else:
			self.batch_count = 0

		path_list = self.data[self.batch_count*batch_size:(self.batch_count+1)*batch_size]

		batch = [get_img(img_path, 128, self.size) for img_path in path_list]
		batch_imgs = np.array(batch).astype(np.float32)
		'''
		print self.batch_count
		fig = self.data2fig(batch_imgs[:16,:,:])
		plt.savefig('out_face/{}.png'.format(str(self.batch_count).zfill(3)), bbox_inches='tight')
		plt.close(fig)
		'''
		return batch_imgs

	def data2fig(self, samples):
		fig = plt.figure(figsize=(4, 4))
		gs = gridspec.GridSpec(4, 4)
		gs.update(wspace=0.05, hspace=0.05)

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample)
		return fig

class mnist():
	def __init__(self, flag='conv', is_tanh = False):
		datapath = prefix + 'mnist'
		self.X_dim = 784 # for mlp
		self.z_dim = 100
		self.y_dim = 10
		self.size = 28 # for conv
		self.channel = 1 # for conv
		self.data = input_data.read_data_sets(datapath, one_hot=True)
		self.flag = flag
		self.is_tanh = is_tanh

	def __call__(self,batch_size):
		batch_imgs,y = self.data.train.next_batch(batch_size)
		if self.flag == 'conv':
			batch_imgs = np.reshape(batch_imgs, (batch_size, self.size, self.size, self.channel)) 
		if self.is_tanh:
			batch_imgs = batch_imgs*2 - 1		
		return batch_imgs, y

	def data2fig(self, samples):
		if self.is_tanh:
			samples = (samples + 1)/2
		fig = plt.figure(figsize=(4, 4))
		gs = gridspec.GridSpec(4, 4)
		gs.update(wspace=0.05, hspace=0.05)

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample.reshape(self.size,self.size), cmap='Greys_r')
		return fig	

if __name__ == '__main__':
	data = face3D()
	print data(17).shape
