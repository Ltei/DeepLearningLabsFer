import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils import tile_raster_images
import matplotlib.pyplot as plt
import math


def debug_ndarray(name, array):
    print(name, " : ", array.shape, " ", np.min(array), " ", np.average(array), " ", np.max(array))


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])
n_samples = mnist.train.num_examples

# training parameters
batch_size = 100
lr = 0.0002
n_epochs = 20


def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)


# D(x)
def discriminator(x, is_train=True, reuse=False):
    assert x.shape[1:] == (32, 32, 1)

    with tf.variable_scope('discriminator', reuse=reuse):
        # 1st hidden layer
        conv = tf.layers.conv2d(x, 64, [4, 4], strides=(2, 2), padding='same')
        lrelu_ = lrelu(conv, 0.2)

        # 2nd hidden layer
        conv = tf.layers.conv2d(lrelu_, 128, [4, 4], strides=(2, 2), padding='same')
        lrelu_ = lrelu(conv, 0.2)

        # output layer
        conv = tf.layers.conv2d(lrelu_, 1, [8, 8], strides=(8, 8), padding='same')
        out = tf.sigmoid(conv)

        assert out.shape[1:] == (1, 1, 1)
        return out, conv


# G(z)
def generator(z, is_train=True):
    assert z.shape[1:] == (1, 1, 100)

    with tf.variable_scope('generator'):
        # 1st hidden layer
        conv = tf.layers.conv2d_transpose(z, 512, [4, 4], strides=(1, 1), padding='valid')
        lrelu_ = lrelu(tf.layers.batch_normalization(conv, training=is_train))

        # 2nd hidden layer
        conv = tf.layers.conv2d_transpose(lrelu_, 256, [4, 4], strides=(2, 2), padding='same')
        lrelu_ = lrelu(tf.layers.batch_normalization(conv, training=is_train))

        # 3rd hidden layer
        conv = tf.layers.conv2d_transpose(lrelu_, 128, [4, 4], strides=(2, 2), padding='same')
        lrelu_ = lrelu(tf.layers.batch_normalization(conv, training=is_train))

        # output layer
        conv = tf.layers.conv2d_transpose(lrelu_, 1, [4, 4], strides=(2, 2), padding='same')
        out = tf.tanh(tf.layers.batch_normalization(conv, training=is_train))

        assert out.shape[1:] == (32, 32, 1)
        return out


def show_generated(G, N, shape=(32, 32), stat_shape=(10, 10), interpolation="bilinear"):
    """Visualization of generated samples
     G - generated samples
     N - number of samples
     shape - dimensions of samples eg (32,32)
     stat_shape - dimension for 2D sample display (eg for 100 samples (10,10)
    """

    image = (tile_raster_images(
        X=G,
        img_shape=shape,
        tile_shape=(int(math.ceil(N / stat_shape[0])), stat_shape[0]),
        tile_spacing=(1, 1)))
    plt.figure(figsize=(10, 14))
    plt.imshow(image, interpolation=interpolation)
    plt.axis('off')
    plt.show()


def save_generated(name, G, N, shape=(32, 32), stat_shape=(10, 10), interpolation="bilinear"):
    """Visualization of generated samples
     G - generated samples
     N - number of samples
     shape - dimensions of samples eg (32,32)
     stat_shape - dimension for 2D sample display (eg for 100 samples (10,10)
    """

    image = (tile_raster_images(
        X=G,
        img_shape=shape,
        tile_shape=(int(math.ceil(N / stat_shape[0])), stat_shape[0]),
        tile_spacing=(1, 1)))
    plt.figure(figsize=(10, 14))
    plt.imshow(image, interpolation=interpolation)
    plt.axis('off')
    plt.savefig(name+str(".jpg"))


def gen_z(N, batch_size):
    z = np.random.normal(0, 1, (batch_size, 1, 1, N))
    return z


# input variables
x = tf.placeholder(tf.float32, shape=(None, 32, 32, 1))
z = tf.placeholder(tf.float32, shape=(None, 1, 1, 100))
isTrain = tf.placeholder(dtype=tf.bool)

# generator
G_z = generator(z, isTrain)

# discriminator
# real
D_real, D_real_logits = discriminator(x, isTrain)
# fake
D_fake, D_fake_logits = discriminator(G_z, isTrain, reuse=True)

# labels for learning
true_labels = tf.ones([batch_size, 1, 1, 1])
fake_labels = tf.zeros([batch_size, 1, 1, 1])

# loss for each network
D_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits,
                                                      labels=true_labels)
D_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits,
                                                      labels=fake_labels)
D_loss = D_loss_real + D_loss_fake
print(D_loss.shape)
G_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits,
                                                 labels=true_labels)
print(G_loss.shape)

# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

# optimizer for each network
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_optim = tf.train.AdamOptimizer(lr, beta1=0.3).minimize(D_loss, var_list=D_vars)
G_optim = tf.train.AdamOptimizer(lr, beta1=0.3).minimize(G_loss, var_list=G_vars)

# open session and initialize all variables
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
tf.global_variables_initializer().run()

# MNIST resize and normalization
train_set = tf.image.resize_images(mnist.train.images, [32, 32]).eval()
# input normalization
debug_ndarray("train_set", train_set)  # TODO

# fixed_z_ = np.random.uniform(-1, 1, (100, 1, 1, 100))
fixed_z_ = gen_z(100, 100)
total_batch = int(n_samples / batch_size)

for epoch in range(n_epochs):

    for iter in range(total_batch):
        # update discriminator
        x_ = train_set[iter * batch_size:(iter + 1) * batch_size]

        # update discriminator
        z_ = gen_z(100, batch_size)
        loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, isTrain: True})

        # update generator
        loss_g_, _ = sess.run([G_loss, G_optim], {z: z_, isTrain: True})

    print('[%d/%d] loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), n_epochs, np.average(loss_d_), np.average(loss_g_)))

    test_images = sess.run(G_z, {z: fixed_z_, isTrain: False})
    save_generated("task5_output_epoch"+str(epoch), test_images, 100)