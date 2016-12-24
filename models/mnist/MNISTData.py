import os, math
import numpy as np
from sklearn.utils import shuffle

raw_training_file = "train.csv"
np_training_data_file = "train_data.npy"
np_training_labels_file = "train_labels.npy"
DELIMITER = ","

# replicates tensorflow.examples.tutorials.mnist for fun
class MNISTData(object):
	"""docstring for MNISTData"""
	def __init__(self, data_dir, one_hot=False, force_pickle=False, proportions=[0.7,0.15,0.15]):
		super(MNISTData, self).__init__()
		self.data_dir = data_dir
		self.one_hot = one_hot

		# load data into (train, cv, test)
		assert(len(proportions) == 3) # (train,cv,test)
		proportions = [p/sum(proportions) for p in proportions] # normalize

		data = []
		labels = []

		extractor = lambda row: (list(map(int,row[1:])), int(row[0]))

		saved_data_file = os.path.join(self.data_dir, np_training_data_file)
		saved_labels_file = os.path.join(self.data_dir, np_training_labels_file)
		if os.path.isfile(saved_data_file) and not force_pickle:
			# load pickled data
			with open(saved_data_file, "rb") as f:
				data = np.load(f)
			with open(saved_labels_file, "rb") as f:
				labels = np.load(f)
		else:
			# create pickled data
			raw_file = os.path.join(data_dir, raw_training_file)
			if not os.path.isfile(raw_file):
				print("Could not find data at %s" % (raw_file))
				return None
			with open(raw_file, "r") as f:
				# first row are headers
				rows = f.readlines()[1:]
				data, labels = zip(*[extractor(row.split(DELIMITER)) for row in rows])
				data, labels = np.array(data, np.uint8), np.array(labels, np.uint8)
				# save pickle
				with open(saved_data_file, "wb") as npf:
					np.save(npf, data)
				with open(saved_labels_file, "wb") as npf:
					np.save(npf, labels)

		assert(len(data) == len(labels))
		# convert to one_hot if necessary
		if self.one_hot:
			labels = convert_to_one_hot(labels, 10)
		(data, labels) = shuffle(data, labels, random_state=0)
		num_train = math.ceil(proportions[0] * len(data))
		last_cv = num_train + math.ceil(proportions[1] * len(data))

		self.train = DataSet(data[:num_train], labels[:num_train])
		self.cv = DataSet(data[num_train:last_cv], labels[num_train:last_cv])
		self.test = DataSet(data[last_cv:], labels[last_cv:])
		
class DataSet(object):
	def __init__(self, data, labels):
		super(DataSet, self).__init__()
		assert(len(data) == len(labels))
		self.data, self.labels = shuffle(data, labels, random_state=0)
		self.data = np.multiply(self.data, 1.0/255.0)
		self.current_pos = 0
		self.num_examples = len(data)

	def next_batch(self, batch_size):
		start = self.current_pos
		self.current_pos += batch_size
		if self.current_pos > self.num_examples:
			# Shuffle the data
			perm = np.arange(self.num_examples)
			np.random.shuffle(perm)
			self.images = self.data[perm]
			self.labels = self.labels[perm]
			# Start next epoch
			start = 0
			self.current_pos = batch_size
		end = self.current_pos
		return self.data[start:end], self.labels[start:end]

def convert_to_one_hot(labels, N):
	assert(labels.ndim == 1)
	assert(max(labels) < N)
	one_hot = np.zeros((labels.size, N))
	one_hot[np.arange(labels.size), labels] = 1
	return one_hot