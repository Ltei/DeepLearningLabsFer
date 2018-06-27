import time
import os
import math
import numpy as np
import tensorflow as tf
import skimage as ski
import skimage.io
from tensorflow.examples.tutorials.mnist import input_data


DATA_DIR = './MNIST/'
SAVE_DIR = "./out_MNIST/"
WEIGHT_DECAY = 1e-4
NB_ITERATIONS = 8
BATCH_SIZE = 50
LR_POLICY = {1:{'lr':1e-3}, 3:{'lr':1e-4}, 5:{'lr':1e-5}, 6:{'lr':1e-6}}


np.random.seed(int(time.time() * 1e6) % 2**31)
dataset = input_data.read_data_sets(DATA_DIR, one_hot=True)
train_x = dataset.train.images
train_x = train_x.reshape([-1, 1, 28, 28])
train_y = dataset.train.labels
valid_x = dataset.validation.images
valid_x = valid_x.reshape([-1, 1, 28, 28])
valid_y = dataset.validation.labels
test_x = dataset.test.images
test_x = test_x.reshape([-1, 1, 28, 28])
test_y = dataset.test.labels
train_mean = train_x.mean()
train_x -= train_mean
valid_x -= train_mean
test_x -= train_mean



def draw_conv_filters(session, layer, epoch, step, name, save_dir):
    weights = session.run(layer).copy()
    num_filters = weights.shape[3]
    num_channels = weights.shape[2]
    k = weights.shape[0]
    assert weights.shape[0] == weights.shape[1]
    weights -= weights.min()
    weights /= weights.max()
    border = 1
    cols = 8
    rows = math.ceil(num_filters / cols)
    width = cols * k + (cols-1) * border
    height = rows * k + (rows-1) * border
    img = np.zeros([height, width, num_channels])
    for i in range(num_filters):
        r = int(i / cols) * (k + border)
        c = int(i % cols) * (k + border)
        img[r:r+k,c:c+k,:] = weights[:,:,:,i]

    img = img.reshape(height, width)
    filename = '%s_epoch_%02d_step_%06d.png' % (name, epoch, step)
    ski.io.imsave(os.path.join(save_dir, filename), img)


# Define network

x = tf.placeholder(tf.float32, [None, 1, 28, 28])
y_ = tf.placeholder(tf.float32, [None, 10])
learning_rate = tf.placeholder(tf.float32)

layer = tf.reshape(x, [-1, 28, 28, 1])

layer = tf.layers.conv2d(layer, 32, 5,
                         padding="same",
                         activation=tf.nn.relu,
                         kernel_initializer=tf.initializers.variance_scaling(),
                         activity_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY),
                         name="conv1")

layer = tf.layers.max_pooling2d(layer, 2, 1, padding="same", name="pool1")

layer = tf.layers.conv2d(layer, 32, 5,
                         padding="same",
                         activation=tf.nn.relu,
                         kernel_initializer=tf.initializers.variance_scaling(),
                         activity_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY),
                         name="conv2")

layer = tf.layers.max_pooling2d(layer, 2, 1, padding="same", name="pool2")

layer = tf.contrib.layers.flatten(layer)

layer = tf.layers.dense(layer, 512,
                        activation=tf.nn.relu,
                        kernel_initializer=tf.initializers.variance_scaling(),
                        activity_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY),
                        name="fc3")

layer = tf.layers.dense(layer, 10,
                        activation=None,
                        kernel_initializer=tf.initializers.variance_scaling(),
                        name="logits")


all_params = tf.trainable_variables()

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=layer))
loss = loss + sum( [tf.nn.l2_loss(param) for param in all_params] )
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
_, accuracy = tf.metrics.accuracy(labels=tf.argmax(y_, 1), predictions=tf.argmax(layer, 1))


num_examples = train_x.shape[0]
assert num_examples % BATCH_SIZE == 0
num_batches = num_examples // BATCH_SIZE
print_every = 20
draw_every = 100



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    print("Iteration 0, Loss = ",loss.eval(feed_dict={x: valid_x, y_: valid_y}),", Accuracy = ",accuracy.eval(feed_dict={x: valid_x, y_: valid_y}))

    for iteration in range(1, NB_ITERATIONS + 1):

        if iteration in LR_POLICY:
            lr_policy = LR_POLICY[iteration]['lr']

        indices = np.arange(num_examples)
        np.random.shuffle(indices)

        for batch_idx in range(num_batches):
            batch_start_idx = batch_idx * BATCH_SIZE
            batch_indices = indices[batch_start_idx : batch_start_idx+BATCH_SIZE]
            batch_x = train_x[batch_indices]
            batch_y_ = train_y[batch_indices]
            train_step.run(session=sess, feed_dict={x: batch_x, y_: batch_y_, learning_rate: lr_policy})
            if batch_idx % print_every == 0:
                batch_loss = loss.eval(feed_dict={x: batch_x, y_: batch_y_})
                print("Iteration ",iteration,", Step ",(batch_idx+print_every),"/",num_batches,", Batch loss = ",batch_loss, end="\r", flush=True)
            if batch_idx % draw_every == 0:
                draw_conv_filters(sess, all_params[0], iteration, batch_idx, "conv1", SAVE_DIR)

        print("Iteration ",iteration,", Loss = ",loss.eval(feed_dict={x: valid_x, y_: valid_y}),", Accuracy = ",accuracy.eval(feed_dict={x: valid_x, y_: valid_y}))

    test_acc = accuracy.eval(feed_dict={x: test_x, y_: test_y})
    test_loss = loss.eval(feed_dict={x: test_x, y_: test_y})
    print('test loss %g, test accuracy %g' % (test_loss, test_acc))
