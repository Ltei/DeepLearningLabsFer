import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from utils import tile_raster_images
import math
import matplotlib.pyplot as plt


def sample_prob(probs):
    """Sample vector x by probability vector p (x = 1) = probs"""
    return tf.to_float(tf.random_uniform(tf.shape(probs)) <= probs)


class RBM:

    def __init__(self, input_rbm=None, v_dim=784, h_dim=100, gibbs_sampling_steps=1, alpha=0.1, init_stddev=0.1):
        self.v_dim = v_dim
        self.h_dim = h_dim
        self.gibbs_sampling_steps = gibbs_sampling_steps
        self.alpha = alpha

        if input_rbm is None:
            self.x = tf.placeholder("float", [None, 784])
            self.v_bias = tf.Variable(tf.zeros([v_dim], tf.float32))
        else:
            self.x = input_rbm.h0_prob
            self.v_bias = input_rbm.h_bias

        self.w = tf.Variable(tf.truncated_normal([v_dim, h_dim], stddev=init_stddev))
        self.h_bias = tf.Variable(tf.zeros([h_dim], tf.float32))

        assert self.x.shape[1:] == v_dim

        self.h0_prob, self.h0 = self.forward_v_to_h(self.x)

        self.h1 = self.h0
        for step in range(gibbs_sampling_steps):
            self.v1_prob, self.v1 = self.forward_h_to_v(self.h1)
            self.h1_prob, self.h1 = self.forward_v_to_h(self.v1)

        self.w_positive_grad = tf.matmul(tf.transpose(self.x), self.h0_prob)
        self.w_negative_grad = tf.matmul(tf.transpose(self.v1_prob), self.h1_prob)
        self.dw = (self.w_positive_grad - self.w_negative_grad) / tf.to_float(tf.shape(self.x)[0])

        self.update_w = tf.assign_add(self.w, alpha * self.dw)
        self.update_v_bias = tf.assign_add(self.v_bias, alpha * tf.reduce_mean(self.x - self.v1, 0))
        self.update_h_bias = tf.assign_add(self.h_bias, alpha * tf.reduce_mean(self.h0 - self.h1, 0))
        self.update_all = (self.update_w, self.update_v_bias, self.update_h_bias)

        v_prob, v = self.forward_h_to_v(self.h1)
        self.err = self.x - v_prob
        self.err_sum = tf.reduce_mean(self.err * self.err)
        self.initialize = tf.global_variables_initializer()

        # Create recs
        _, self.rec = self.forward_h_to_v(self.h0)
        if input_rbm is not None:
            input_rbm.rec, _ = input_rbm.forward_h_to_v(self.rec)

    def forward_v_to_h(self, v):
        assert v.shape[1:] == self.v_dim
        prob = tf.nn.sigmoid(tf.matmul(v, self.w) + self.h_bias)
        return prob, sample_prob(prob)

    def forward_h_to_v(self, h):
        assert h.shape[1:] == self.h_dim
        prob = tf.nn.sigmoid(tf.matmul(h, tf.transpose(self.w)) + self.v_bias)
        return prob, sample_prob(prob)


def draw_weights(W, shape, N):
    """Visualization of weight
     W - weight vector
     shape - tuple dimensions for 2D weight display - usually input image dimensions, eg (28,28)
     N - number weight vectors
     shape_state - Dimension for 2D state display (eg for 100 states (10,10)
    """
    image = (tile_raster_images(
        X=W.T,
        img_shape=shape,
        tile_shape=(int(math.ceil(N / 20)), 20),
        tile_spacing=(1, 1)))
    plt.figure(figsize=(10, 14))
    plt.imshow(image, interpolation="bilinear")
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


def reconstruct(ind, states, orig, weights, biases, v_shape, h_shape, h_dim):
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

    draw_rec(states[ind], 'states', h_shape, Nrows, in_a_row, j)
    j += 1
    draw_rec(orig[ind], 'input', v_shape, Nrows, in_a_row, j)

    reconstr = biases.copy()
    j += 1
    draw_rec(sigmoid(reconstr), 'biases', v_shape, Nrows, in_a_row, j)

    for i in range(h_dim):
        if states[ind, i] > 0:
            j += 1
            reconstr = reconstr + weights[:, i]
            titl = '+= s' + str(i + 1)
            draw_rec(sigmoid(reconstr), titl, v_shape, Nrows, in_a_row, j)
    plt.tight_layout()
