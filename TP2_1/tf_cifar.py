import time
import os
import math
import pickle
import skimage.io
import numpy as np
import tensorflow as tf
import skimage as ski
import matplotlib.pyplot as plt

DATA_DIR = './cifar/'
SAVE_DIR = "./out_cifar/"
IMG_HEIGHT = 32
IMG_WIDTH = 32
NUM_CHANNELS = 3
NUM_CLASSES = 10
BATCH_SIZE = 256
MAX_EPOCHS = 100
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4

## Utils

def to_one_hot(class_value, class_count=10):
    result = [0] * class_count
    result[class_value] = 1
    return result


## Dataset

def shuffle_data(data_x, data_y):
  indices = np.arange(data_x.shape[0])
  np.random.shuffle(indices)
  shuffled_data_x = np.ascontiguousarray(data_x[indices])
  shuffled_data_y = np.ascontiguousarray(data_y[indices])
  return shuffled_data_x, shuffled_data_y

def unpickle(file):
  fo = open(file, 'rb')
  dict = pickle.load(fo, encoding='latin1')
  fo.close()
  return dict


train_x = np.ndarray((0, IMG_HEIGHT * IMG_WIDTH * NUM_CHANNELS), dtype=np.float32)
train_y = []
for i in range(1, 6):
  subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
  train_x = np.vstack((train_x, subset['data']))
  train_y += subset['labels']
train_x = train_x.reshape((-1, NUM_CHANNELS, IMG_HEIGHT, IMG_WIDTH)).transpose(0,2,3,1)
train_y = np.array(train_y, dtype=np.int32)

subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
test_x = subset['data'].reshape((-1, NUM_CHANNELS, IMG_HEIGHT, IMG_WIDTH)).transpose(0,2,3,1).astype(np.float32)
test_y = np.array(subset['labels'], dtype=np.int32)

valid_size = 5000
train_x, train_y = shuffle_data(train_x, train_y)
valid_x = train_x[:valid_size, ...]
valid_y = train_y[:valid_size, ...]
train_x = train_x[valid_size:, ...]
train_y = train_y[valid_size:, ...]
data_mean = train_x.mean((0,1,2))
data_std = train_x.std((0,1,2))

train_x = (train_x - data_mean) / data_std
valid_x = (valid_x - data_mean) / data_std
test_x = (test_x - data_mean) / data_std

# Add dimension for broadcasting
train_x = train_x.reshape([-1, IMG_WIDTH, IMG_HEIGHT, NUM_CHANNELS])
valid_x = valid_x.reshape([-1, IMG_WIDTH, IMG_HEIGHT, NUM_CHANNELS])
test_x = test_x.reshape([-1, IMG_WIDTH, IMG_HEIGHT, NUM_CHANNELS])

# Convert labels to one hot
train_y = np.array(list(map(to_one_hot, train_y)))
valid_y = np.array(list(map(to_one_hot, valid_y)))
test_y = np.array(list(map(to_one_hot, test_y)))


## Draw filters

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
    filename = '%s_epoch_%02d_step_%06d.png' % (name, epoch, step)
    ski.io.imsave(os.path.join(save_dir, filename), img)



## Graph

graph_x = tf.placeholder(tf.float32, [None, 32, 32, 3])
graph_y_ = tf.placeholder(tf.float32, [None, 10])

graph_conv1 = tf.layers.conv2d(graph_x, 16, 5,
                         padding="same",
                         activation=tf.nn.relu,
                         kernel_initializer=tf.initializers.variance_scaling(),
                         activity_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY),
                         name="conv1")

print("graph_conv1 : ",graph_conv1.shape)

graph_pool1 = tf.layers.max_pooling2d(graph_conv1, 3, 2, padding="same", name="pool1")

print("graph_pool1 : ",graph_pool1.shape)

graph_conv2 = tf.layers.conv2d(graph_pool1, 32, 5,
                         padding="same",
                         activation=tf.nn.relu,
                         kernel_initializer=tf.initializers.variance_scaling(),
                         activity_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY),
                         name="conv2")

print("graph_conv2 : ",graph_conv2.shape)

graph_pool2 = tf.layers.max_pooling2d(graph_conv2, 3, 2, padding="same", name="pool2")

print("graph_pool2 : ",graph_pool2.shape)

graph_pool2_reshaped = tf.reshape(graph_pool2, [-1, 8*8*32])

print("graph_pool2_reshaped : ",graph_pool2_reshaped.shape)

graph_fc1 = tf.layers.dense(graph_pool2_reshaped, 256,
                     activation=tf.nn.relu,
                     kernel_initializer=tf.initializers.variance_scaling(),
                     activity_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY),
                     name="fc1")

print("graph_fc1 : ",graph_fc1.shape)

graph_fc2 = tf.layers.dense(graph_fc1, 128,
                     activation=tf.nn.relu,
                     kernel_initializer=tf.initializers.variance_scaling(),
                     activity_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY),
                     name="fc2")

print("graph_fc2 : ",graph_fc2.shape)

graph_y = tf.layers.dense(graph_fc2, 10,
                    kernel_initializer=tf.initializers.variance_scaling(),
                    activity_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY),
                    name="fc3")

print("graph_y : ",graph_y.shape)

graph_weights = []
for var in tf.trainable_variables():
    if "/kernel:0" not in var.name:
        continue
    graph_weights.append(var)

graph_err_loss = graph_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=graph_y_, logits=graph_y))
graph_reg_loss = graph_loss + sum(map(lambda w : tf.nn.l2_loss(w), graph_weights))
graph_loss = graph_err_loss + WEIGHT_DECAY * graph_reg_loss

graph_global_step = tf.Variable(0, trainable=False)
graph_learning_rate = tf.train.exponential_decay(LEARNING_RATE, graph_global_step, 500, 0.95, staircase=True)
graph_train_step = tf.train.MomentumOptimizer(graph_learning_rate, 0.9).minimize(graph_loss, global_step=graph_global_step)

_, graph_accuracy = tf.metrics.accuracy(labels=tf.argmax(graph_y_, 1), predictions=tf.argmax(graph_y, 1))


## Draw efficiency graph

def plot_training_progress(save_dir, data):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,8))

    linewidth = 2
    legend_size = 10
    train_color = 'm'
    val_color = 'c'

    num_points = len(data['train_loss'])
    x_data = np.linspace(1, num_points, num_points)
    ax1.set_title('Cross-entropy loss')
    ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
    ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
    ax1.legend(loc='upper right', fontsize=legend_size)
    ax2.set_title('Average class accuracy')
    ax2.plot(x_data, data['train_acc'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
    ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
    ax2.legend(loc='upper left', fontsize=legend_size)
    ax3.set_title('Learning rate')
    ax3.plot(x_data, data['lr'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='learning_rate')
    ax3.legend(loc='upper left', fontsize=legend_size)

    save_path = os.path.join(save_dir, 'training_plot.pdf')
    print('Plotting in: ', save_path)
    plt.savefig(save_path)

plot_data = {}
plot_data['train_loss'] = []
plot_data['valid_loss'] = []
plot_data['train_acc'] = []
plot_data['valid_acc'] = []
plot_data['lr'] = []

## Training

num_examples = train_x.shape[0]
num_batches = num_examples // BATCH_SIZE
print_every = 5
draw_every = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    print("train_x.shape = ",train_x.shape)
    print("train_y.shape = ",train_y.shape)

    for epoch in range(1, MAX_EPOCHS + 1):

        indices = np.arange(num_examples)
        np.random.shuffle(indices)

        for batch_idx in range(num_batches):
            batch_start_idx = batch_idx * BATCH_SIZE
            batch_indices = indices[batch_start_idx : batch_start_idx+BATCH_SIZE]
            batch_x = train_x[batch_indices]
            batch_y_ = train_y[batch_indices]
            graph_train_step.run(session=sess, feed_dict={graph_x: batch_x, graph_y_: batch_y_})
            if batch_idx % print_every == 0:
                batch_loss = graph_loss.eval(feed_dict={graph_x: batch_x, graph_y_: batch_y_})
                print('Epoch %d :  Step %d/%d, Batch loss %g, Learning rate: %g' % (epoch, batch_idx, num_batches, batch_loss, sess.run(graph_learning_rate)), end="\r", flush=True)
            if batch_idx % draw_every == 0:
                draw_conv_filters(sess, graph_weights[0], epoch, batch_idx, "conv1", SAVE_DIR)

        print("", flush=True)
        valid_acc = graph_accuracy.eval(feed_dict={graph_x: valid_x, graph_y_: valid_y})
        valid_loss = graph_loss.eval(feed_dict={graph_x: valid_x, graph_y_: valid_y})
        print('epoch %d, valid loss %g, valid accuracy %g' % (epoch, valid_loss, valid_acc))
        train_acc = graph_accuracy.eval(feed_dict={graph_x: test_x, graph_y_: test_y})
        train_loss = graph_loss.eval(feed_dict={graph_x: test_x, graph_y_: test_y})

        plot_data['train_loss'] += [train_loss]
        plot_data['valid_loss'] += [valid_loss]
        plot_data['train_acc'] += [train_acc]
        plot_data['valid_acc'] += [valid_acc]
        plot_data['lr'] += [graph_learning_rate.eval(session=sess)]
        plot_training_progress(SAVE_DIR, plot_data)

    test_acc = graph_accuracy.eval(feed_dict={graph_x: test_x, graph_y_: test_y})
    test_loss = graph_loss.eval(feed_dict={graph_x: test_x, graph_y_: test_y})
    print('test loss %g, test accuracy %g' % (test_loss, test_acc))
