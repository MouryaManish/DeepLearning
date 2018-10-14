import numpy as np
import random
import sys
sys.path.append("/home/manish.mourya001/fileStack/assignment1/cs231n")
from data_utils import load_CIFAR10
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from classifiers.k_nearest_neighbor import KNearestNeighbor

plt.rcParams['figure.figsize']=(10.0,8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

cifar10_dir = '/home/manish.mourya001/fileStack/assignment1/cs231n/datasets/cifar-10-batches-py'

try:
# previous data cleared if any
    del X_train,y_train
    del X_test, y_test
    print('old data cleared')
except:
    print('error from clearing the  previous data.... this is ok')



X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
"""classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
with PdfPages('test.pdf') as pdf:
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
#            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            fig = plt.figure()
            pdf.savefig(fig)
            if i == 0:
                plt.title(cls)
"""
"""
# making subsample
num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]
"""


classifier = KNearestNeighbor()
classifier.train(X_train,y_train)
dists = classifier.compute_distances_two_loops(X_test)


# Now implement the function predict_labels and run the code below:
# We use k = 1 (which is Nearest Neighbor).
y_test_pred = classifier.predict_labels(dists, k=1)

# Compute and print the fraction of correctly predicted examples
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

y_test_pred = classifier.predict_labels(dists, k=5)
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))


# Now lets speed up distance matrix computation by using partial vectorization
# with one loop. Implement the function compute_distances_one_loop and run the
# code below:
dists_one = classifier.compute_distances_one_loop(X_test)

 
