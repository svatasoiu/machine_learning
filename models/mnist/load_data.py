import os, math
import numpy as np
from sklearn.utils import shuffle

raw_training_file = "data/train.csv"
np_training_data_file = "data/train_data.npy"
np_training_labels_file = "data/train_labels.npy"
DELIMITER = ","

# array of (feature vector, label) tuples
def load_mnist_data(force_pickle=False):
	data = []
	labels = []

	training_extractor = lambda row: (list(map(int,row[1:])), int(row[0]))

	if os.path.isfile(np_training_data_file) and not force_pickle:
		# load pickled data
		with open(np_training_data_file, "rb") as f:
			data = np.load(f)
		with open(np_training_labels_file, "rb") as f:
			labels = np.load(f)
	else:
		# create pickled data
		if not os.path.isfile(raw_training_file):
			print("Could not find data at %s" % (raw_file))
			return None
		with open(raw_training_file, "r") as f:
			# first row are headers
			rows = f.readlines()[1:]
			data, labels = zip(*[training_extractor(row.split(DELIMITER)) for row in rows])
			data, labels = np.array(data, np.uint8), np.array(labels, np.uint8)
			# save pickle
			with open(np_training_data_file, "wb") as npf:
				np.save(npf, data)
			with open(np_training_labels_file, "wb") as npf:
				np.save(npf, labels)

	assert(len(data) == len(labels))
	return (data, labels)

def partition_data(data, labels, proportions=[0.7,0.15,0.15]):
	assert(len(data) == len(labels))
	assert(len(proportions) == 3) # (train,cv,test)
	proportions = [p/sum(proportions) for p in proportions] # normalize

	(data, labels) = shuffle(data, labels, random_state=0)
	num_train = math.ceil(proportions[0] * len(data))
	last_cv = num_train + math.ceil(proportions[1] * len(data))

	return ((data[:num_train],labels[:num_train]),
		(data[num_train:last_cv],labels[num_train:last_cv]),
		(data[last_cv:], labels[last_cv:]))