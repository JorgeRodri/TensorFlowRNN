from TFmodels.rbm_after_refactor import RBM
import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data


old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_data = mnist.train.images  # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images  # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

tf.logging.set_verbosity(old_v)

RBM_visible_sizes = 784
RBM_hidden_sizes = 600

# rbm = RBM(RBM_visible_sizes, RBM_hidden_sizes)
# rbm.fit(train_data)
