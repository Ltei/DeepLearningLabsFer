

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import data



class TFLogreg:

  """
    Arguments:
    - D: dimensions of each datapoint
    - C: number of classes
    - param_delta: training step
  """
  def __init__(self, D, C, param_delta=0.5):
    # declare graph nodes for the data and parameters:
    self.X  = tf.placeholder(tf.float32, [None, D])
    self.Yoh_ = tf.placeholder(tf.float32, [None, C])
    self.W = tf.Variable(tf.random_normal([C, D]))
    self.b = tf.Variable(tf.random_normal([C]))

    # formulate the model: calculate self.probs
    scores = tf.matmul(self.X, self.W) + self.b
    self.probs = tf.nn.softmax(scores)

    # formulate the loss: self.loss
    err_loss = tf.reduce_sum(self.Yoh_ * (-tf.log(self.probs)), 1)
    self.loss = tf.reduce_mean(err_loss)

    # formulate the training operation: self.train_step
    trainer = tf.train.GradientDescentOptimizer(param_delta)
    self.train_step = trainer.minimize(self.loss)

    # instantiate the execution context: self.session
    self.session = tf.Session()

  """
    Arguments:
    - X: actual datapoints [NxD]
    - Yoh_: one-hot encoded labels [NxC]
    - param_niter: number of iterations
  """
  def train(self, X, Yoh_, param_niter, param_print_step=-1):
    if param_print_step < 0:
        param_print_step = param_niter / 10

    # parameter intiailization
    self.session.run(tf.global_variables_initializer())

    # optimization loop
    for i in range(param_niter):
        self.session.run([self.loss, self.train_step], feed_dict={self.X: X, self.Yoh_: Yoh_})
        if i % param_print_step == 0:
            accuracy = data.eval_perf_multi(self.eval(X), Yoh_)
            print("Iteration = ",i,", Loss = ",self.loss,", Accuracy = ",accuracy)

  """
    Arguments:
    - X: actual datapoints [NxD]
    Returns: predicted class probabilites [NxC]
  """
  def eval(self, X):
    return self.session.run(self.probs, feed_dict={self.X: X})



if __name__ == "__main__":
    # initialize the random number generator
    np.random.seed(100)
    tf.set_random_seed(100)

    # instantiate the data X and the labels Yoh_
    X, Y_ = data.sample_gauss_2d(2, 100)
    Yoh_ = data.class_to_one_hot(Y_)

    # build the graph:
    tflr = TFLogreg(X.shape[1], Yoh_.shape[1], 0.5)

    # perform the training with given hyper-parameters:
    tflr.train(X, Yoh_, 1000)

    # predict probabilities of the data points
    probs = tflr.eval(X)
    Y = probs.argmax(axis=1)

    # print performance (per-class precision and recall)

    # draw results, decision surface
    decfun = lambda x: tflr.eval(x).max(axis=1)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)
    data.graph_data(X, Y_, Y, special=[])
    plt.show()
