import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from utils import tile_raster_images
import math
import matplotlib.pyplot as plt

# %matplotlib inline
plt.rcParams['image.cmap'] = 'jet'

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, \
                     mnist.test.labels


def weights(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias(shape):
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial)


def sample_prob(probs):
    """Sample vector x by probability vector p (x = 1) = probs"""
    return tf.to_float(tf.random_uniform(tf.shape(probs)) <= probs)


def draw_weights(W, shape, N, stat_shape, interpolation="bilinear"):
    """Visualization of weight
     W - weight vector
     shape - tuple dimensions for 2D weight display - usually input image dimensions, eg (28,28)
     N - number weight vectors
     shape_state - Dimension for 2D state display (eg for 100 states (10,10)
    """
    image = (tile_raster_images(
        X=W.T,
        img_shape=shape,
        tile_shape=(int(math.ceil(N / stat_shape[0])), stat_shape[0]),
        tile_spacing=(1, 1)))
    plt.figure(figsize=(10, 14))
    plt.imshow(image, interpolation=interpolation)
    plt.axis('off')


def draw_reconstructions(ins, outs, states, shape_in, shape_state, N):
    """Visualization of inputs and associated reconstructions and hidden layer states
     ins -- input vectors
     outs - reconstructed vectors
     states - hidden layer state vectors
     shape_in - dimension of input images eg (28,28)
     shape_state - Dimension for 2D state display (eg for 100 states (10,10)
     N - number of samples
    """
    plt.figure(figsize=(8, int(2 * N)))
    for i in range(N):
        plt.subplot(N, 4, 4 * i + 1)
        plt.imshow(ins[i].reshape(shape_in), vmin=0, vmax=1, interpolation="nearest")
        plt.title("Test input")
        plt.axis('off')
        plt.subplot(N, 4, 4 * i + 2)
        plt.imshow(outs[i][0:784].reshape(shape_in), vmin=0, vmax=1, interpolation="nearest")
        plt.title("Reconstruction")
        plt.axis('off')
        plt.subplot(N, 4, 4 * i + 3)
        plt.imshow(states[i].reshape(shape_state), vmin=0, vmax=1, interpolation="nearest")
        plt.title("States")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def draw_generated(stin, stout, gen, shape_gen, shape_state, N):
    """Visualization of initial hidden states, final hidden states and associated reconstructions
     stin - the initial hidden layer
     stout - reconstructed vectors
     gen - vector of hidden layer state
     shape_gen - dimensional input image eg (28,28)
     shape_state - Dimension for 2D state display (eg for 100 states (10,10)
     N - number of samples
    """
    plt.figure(figsize=(8, int(2 * N)))
    for i in range(N):
        plt.subplot(N, 4, 4 * i + 1)
        plt.imshow(stin[i].reshape(shape_state), vmin=0, vmax=1, interpolation="nearest")
        plt.title("set state")
        plt.axis('off')
        plt.subplot(N, 4, 4 * i + 2)
        plt.imshow(stout[i][0:784].reshape(shape_state), vmin=0, vmax=1, interpolation="nearest")
        plt.title("final state")
        plt.axis('off')
        plt.subplot(N, 4, 4 * i + 3)
        plt.imshow(gen[i].reshape(shape_gen), vmin=0, vmax=1, interpolation="nearest")
        plt.title("generated visible")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


Nh = 100  # The number of elements of the first hidden layer
h1_shape = (10, 10)
Nv = 784  # The number of elements of the first hidden layerBroj elemenata vidljivog sloja
v_shape = (28, 28)
Nu = 5000  # Number of samples for visualization of reconstruction

gibbs_sampling_steps = 1
alpha = 0.1

g1 = tf.Graph()
with g1.as_default():
    X1 = tf.placeholder("float", [None, Nv])
    w1 = weights([Nv, Nh])
    vb1 = bias([Nv])
    hb1 = bias([Nh])

    h0_prob = tf.nn.sigmoid(tf.matmul(X1, w1) + hb1)  # DONE
    h0 = sample_prob(h0_prob)
    h1 = h0

    for step in range(gibbs_sampling_steps):
        v1_prob = tf.nn.sigmoid(tf.matmul(h1, tf.transpose(w1)) + vb1)  # DONE
        v1 = sample_prob(v1_prob)  # DONE
        h1_prob = tf.nn.sigmoid(tf.matmul(v1, w1) + hb1)  # DONE
        h1 = sample_prob(h1_prob)  # DONE

    w1_positive_grad = tf.matmul(tf.transpose(X1), h0_prob)  # DONE
    w1_negative_grad = tf.matmul(tf.transpose(v1_prob), h1_prob)

    dw1 = (w1_positive_grad - w1_negative_grad) / tf.to_float(tf.shape(X1)[0])

    update_w1 = tf.assign_add(w1, alpha * dw1)
    update_vb1 = tf.assign_add(vb1, alpha * tf.reduce_mean(X1 - v1, 0))
    update_hb1 = tf.assign_add(hb1, alpha * tf.reduce_mean(h0 - h1, 0))

    out1 = (update_w1, update_vb1, update_hb1)

    v1_prob = tf.nn.sigmoid(tf.matmul(h1, tf.transpose(w1)) + vb1)  # DONE
    v1 = sample_prob(v1_prob)  # DONE

    err1 = X1 - v1_prob
    err_sum1 = tf.reduce_mean(err1 * err1)

    initialize1 = tf.global_variables_initializer()

batch_size = 100
epochs = 100
n_samples = mnist.train.num_examples
total_batch = int(n_samples / batch_size) * epochs

sess1 = tf.Session(graph=g1)
sess1.run(initialize1)

for i in range(total_batch):
    batch, label = mnist.train.next_batch(batch_size)
    err, _ = sess1.run([err_sum1, out1], feed_dict={X1: batch})

    if i % (int(total_batch / 10)) == 0:
        print("Batch = ", i, "Error = ", err)

w1s = w1.eval(session=sess1)
vb1s = vb1.eval(session=sess1)
hb1s = hb1.eval(session=sess1)
vr, h1s = sess1.run([v1_prob, h1], feed_dict={X1: teX[0:Nu, :]})

# visualization of weights
draw_weights(w1s, v_shape, Nh, h1_shape)

# visualization of reconstructions and states
draw_reconstructions(teX, vr, h1s, v_shape, h1_shape, 150)


# visualization of a reconstructions with the gradual addition of the contributions of active hidden elements
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def draw_rec(inp, title, size, Nrows, in_a_row, j):
    """ Draw an iteration of creating the visible layer
     inp - visible layer
     title - thumbnail title
     size - 2D dimensions of visible layer
     Nrows - max. number of thumbnail rows
     in-a-row. number of thumbnails in one row
     j - position of thumbnails in the grid
    """
    plt.subplot(Nrows, in_a_row, j)
    plt.imshow(inp.reshape(size), vmin=0, vmax=1, interpolation="nearest")
    plt.title(title)
    plt.axis('off')


def reconstruct(ind, states, orig, weights, biases):
    """ Sequential visualization of  the visible layer reconstruction
     ind - index of digits in orig (matrix with digits as lines)
     states - state vectors of input vectors
     orig - original input vectors
     weights - weight matrix
    """
    j = 1
    in_a_row = 6
    Nimg = states.shape[1] + 3
    Nrows = int(np.ceil(float(Nimg + 2) / in_a_row))

    plt.figure(figsize=(12, 2 * Nrows))

    draw_rec(states[ind], 'states', h1_shape, Nrows, in_a_row, j)
    j += 1
    draw_rec(orig[ind], 'input', v_shape, Nrows, in_a_row, j)

    reconstr = biases.copy()
    j += 1
    draw_rec(sigmoid(reconstr), 'biases', v_shape, Nrows, in_a_row, j)

    for i in range(Nh):
        if states[ind, i] > 0:
            j += 1
            reconstr = reconstr + weights[:, i]
            titl = '+= s' + str(i + 1)
            draw_rec(sigmoid(reconstr), titl, v_shape, Nrows, in_a_row, j)
    plt.tight_layout()


reconstruct(0, h1s, teX, w1s, vb1s)  # the first argument is the digit index in the digit matrix

# The probability that the hidden state is included through Nu input samples
plt.figure()
tmp = (h1s.sum(0) / h1s.shape[0]).reshape(h1_shape)
plt.imshow(tmp, vmin=0, vmax=1, interpolation="nearest")
plt.axis('off')
plt.colorbar()
plt.title('likelihood of the activation of certain neurons of the hidden layer')

# Visualization of weights sorted by frequency
tmp_ind = (-tmp).argsort(None)
draw_weights(w1s[:, tmp_ind], v_shape, Nh, h1_shape)
plt.title('Sorted weight matrices - from most to the least used')

# Generating samples from random vectors
r_input = np.random.rand(100, Nh)
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

out_1 = sess1.run((v1), feed_dict={h0: r_input})

# Emulation of additional Gibbs sampling using feed_dict
for i in range(1000):
    out_1_prob, out_1, hout1 = sess1.run((v1_prob, v1, h1), feed_dict={X1: out_1})

draw_generated(r_input, hout1, out_1_prob, v_shape, h1_shape, 200)
