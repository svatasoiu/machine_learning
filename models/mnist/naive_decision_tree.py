import MNISTData
from sklearn import tree, metrics

# get data
print("Loading Data...")
mnist = MNISTData.MNISTData(data_dir="data/")

# create classifier
print("Training Decision Tree Classifier...")
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(mnist.train.data, mnist.train.labels)

# calculate predictions on test data
print("Making Predictions...")
predicted_labels = classifier.predict(mnist.test.data)

# evaluate predictions
print("Evaluating Classifier...")
accuracy = metrics.accuracy_score(mnist.test.labels, predicted_labels)
confusion_matrix = metrics.confusion_matrix(mnist.test.labels, predicted_labels)

print("========Results========")
print("Accuracy: %f" % (accuracy))
print(confusion_matrix)