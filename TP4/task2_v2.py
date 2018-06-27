import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from utils import tile_raster_images
import math
import matplotlib.pyplot as plt
from RBM import RBM, draw_weights, draw_generated, draw_reconstructions, sigmoid, draw_rec, reconstruct

# %matplotlib inline
plt.rcParams['image.cmap'] = 'jet'

#
# INIT
#

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, \
                     mnist.test.labels

V_SHAPE = (28, 28)
V_DIM = 784
H1_SHAPE = (10, 10)
H1_DIM = 100
RECONSTRUCTION_COUNT = 20

batch_size = 100
epochs = 100
n_samples = mnist.train.num_examples
total_batch = int(n_samples / batch_size) * epochs
Nu = 5000  # Number of samples for visualization of reconstruction

with tf.variable_scope("l1_"):
    rbm1 = RBM(v_dim=V_DIM, h_dim=H1_DIM)
with tf.variable_scope("l2_"):
    rbm2 = RBM(rbm1, v_dim=H1_DIM, h_dim=121)

with tf.Session() as sess:

    sess.run(rbm1.initialize)
    sess.run(rbm2.initialize)

    for i in range(total_batch):
        batch, label = mnist.train.next_batch(batch_size)
        err, _ = sess.run([rbm1.err_sum, rbm1.update_all], feed_dict={rbm1.x: batch})

        if i % (int(total_batch / 10)) == 0:
            print("Layer1 => Batch = ", i, "Error = ", err)

    for i in range(total_batch):
        batch, label = mnist.train.next_batch(batch_size)
        err, _ = sess.run([rbm2.err_sum, rbm2.update_all], feed_dict={rbm1.x: batch})

        if i % (int(total_batch / 10)) == 0:
            print("Layer2 => Batch = ", i, "Error = ", err)

    w1s = rbm1.w.eval()
    w2s = rbm2.w.eval()
    vb1s = rbm1.v_bias.eval()
    hb1s = rbm1.h_bias.eval()
    vr, h1s = sess.run([rbm1.rec, rbm1.h0], feed_dict={rbm1.x: teX[:RECONSTRUCTION_COUNT, :]})

# Visualization of weights
draw_weights(w1s, V_SHAPE, H1_DIM)
plt.savefig("task2_weights1")
draw_weights(w2s, (10, 10), 121)
plt.savefig("task2_weights2")

# Visualization of reconstruction and states
draw_reconstructions(teX, vr, h1s, V_SHAPE, H1_SHAPE, RECONSTRUCTION_COUNT)
plt.savefig("task2_reconstruction")