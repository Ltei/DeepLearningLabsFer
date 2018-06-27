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
RECONSTRUCTION_DRAW_COUNT = 20
GENERATED_DRAW_COUNT = 20

batch_size = 100
epochs = 100
n_samples = mnist.train.num_examples
total_batch = int(n_samples / batch_size) * epochs
Nu = 5000  # Number of samples for visualization of reconstruction

rbm1 = RBM(v_dim=V_DIM, h_dim=H1_DIM)

sess = tf.Session()
sess.run(rbm1.initialize)

#
# TRAIN
#

for i in range(total_batch):
    batch, label = mnist.train.next_batch(batch_size)
    err, _ = sess.run([rbm1.err_sum, rbm1.update_all], feed_dict={rbm1.x: batch})

    if i % (int(total_batch / 10)) == 0:
        print("Batch = ", i, "Error = ", err)

w1s = rbm1.w.eval(session=sess)
vb1s = rbm1.v_bias.eval(session=sess)
hb1s = rbm1.h_bias.eval(session=sess)
vr, h1s = sess.run([rbm1.v1_prob, rbm1.h1], feed_dict={rbm1.x: teX[0:RECONSTRUCTION_DRAW_COUNT, :]})

# Visualization of weights
draw_weights(w1s, V_SHAPE, rbm1.h_dim)
# Visualization of reconstructions and states
draw_reconstructions(teX, vr, h1s, V_SHAPE, H1_SHAPE, RECONSTRUCTION_DRAW_COUNT)

#
# SAMPLE
#

# visualization of a reconstructions with the gradual addition of the contributions of active hidden elements
# the first argument is the digit index in the digit matrix
reconstruct(0, h1s, teX, w1s, vb1s, V_SHAPE, H1_SHAPE, H1_DIM)

# The probability that the hidden state is included through Nu input samples
plt.figure()
tmp = (h1s.sum(0) / h1s.shape[0]).reshape(H1_SHAPE)
plt.imshow(tmp, vmin=0, vmax=1, interpolation="nearest")
plt.axis('off')
plt.colorbar()
plt.title('likelihood of the activation of certain neurons of the hidden layer')

# Visualization of weights sorted by frequency
tmp_ind = (-tmp).argsort(None)
draw_weights(w1s[:, tmp_ind], V_SHAPE, H1_DIM)
plt.title('Sorted weight matrices - from most to the least used')

# Generating samples from random vectors
r_input = np.random.rand(100, H1_DIM)
r_input[r_input > 0.9] = 1  # percentage of active - vary freely
r_input[r_input < 1] = 0
r_input = r_input * 20  # Boosting in case a small percentage is active

s = 10
i = 0
r_input[i, :] = 0
r_input[i, i] = s
i += 1
r_input[i, :] = 0
r_input[i, i] = s
i += 1
r_input[i, :] = 0
r_input[i, i] = s
i += 1
r_input[i, :] = 0
r_input[i, i] = s
i += 1
r_input[i, :] = 0
r_input[i, i] = s
i += 1
r_input[i, :] = 0
r_input[i, i] = s
i += 1
r_input[i, :] = 0
r_input[i, i] = s

out_1 = sess.run(rbm1.v1, feed_dict={rbm1.h0: r_input})

# Emulation of additional Gibbs sampling using feed_dict
for i in range(1000):
    out_1_prob, out_1, hout1 = sess.run([rbm1.v1_prob, rbm1.v1, rbm1.h1], feed_dict={rbm1.x: out_1})

draw_generated(r_input, hout1, out_1_prob, V_SHAPE, H1_SHAPE, GENERATED_DRAW_COUNT)
