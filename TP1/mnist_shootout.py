

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tf_deep import TFDeep
import data
import numpy as np

tf.app.flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')
tf.set_random_seed(100)
np.random.seed(100)
mnist = input_data.read_data_sets(tf.app.flags.FLAGS.data_dir, one_hot=True)

N=mnist.train.images.shape[0]
D=mnist.train.images.shape[1]
C=mnist.train.labels.shape[1]

network = TFDeep([D, C], param_delta=0.5)

print(":::::::::::::::::::::::")
print(":::::::::::::::::::::::")
print(":::::::::::::::::::::::")
network.train_batch(mnist.train.images, mnist.train.labels, 500, param_print_step=25, param_batch_size=1000)
print(":::::::::::::::::::::::")
print(":::::::::::::::::::::::")
print(":::::::::::::::::::::::")

# predict probabilities of the data points
probs = network.eval(mnist.test.images)
Y = probs.argmax(axis=1)
Y_ = mnist.test.labels.argmax(axis=1)

# print performance (per-class precision and recall)
accuracy, _, _ = data.eval_perf_multi(Y_, Y)
print("Accuracy = ",accuracy)
