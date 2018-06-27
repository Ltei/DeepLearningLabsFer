

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import data



class TFDeep:
    def __init__(self, layers, param_delta=0.5):
        # declare graph nodes for the data and parameters:
        last_idx = len(layers) - 1
        self.X  = tf.placeholder(tf.float32, [None, layers[0]])

        self.Ws = []
        self.bs = []
        for i in range(0, last_idx):
            self.Ws.append( tf.Variable(tf.random_normal( [layers[i], layers[i+1]] ), name="weights_matrix_"+str(i) ) )
            self.bs.append( tf.Variable(tf.random_normal( [layers[i+1]] ), name="bias_vector_"+str(i) ) )

        self.Yoh_ = tf.placeholder(tf.float32, [None, layers[-1]])

        tmp = self.X
        last_idx = len(self.Ws) - 1
        for i in range(last_idx):
            tmp = tf.nn.tanh(tf.matmul(tmp, self.Ws[i]) + self.bs[i])
        self.probs = tf.nn.softmax(tf.matmul(tmp, self.Ws[last_idx]) + self.bs[last_idx])

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
            loss, _ = self.session.run([self.loss, self.train_step], feed_dict={self.X: X, self.Yoh_: Yoh_})
            if i % param_print_step == 0:
                print("Iteration = ",i,", Loss = ",loss)

    def train_batch(self, X, Yoh_, param_niter, param_print_step=-1, param_batch_size=1000):
        print("len(X) = ",len(X))
        if param_print_step < 0:
            param_print_step = param_niter / 10

        # parameter intiailization
        self.session.run(tf.global_variables_initializer())

        # optimization loop
        for i in range(param_niter):

            for j in range(0, X.shape[0], param_batch_size):
                if j+param_batch_size > len(X):
                    loss, _ = self.session.run([self.loss, self.train_step], feed_dict={self.X: X[j:], self.Yoh_: Yoh_[j:]})
                else:
                    loss, _ = self.session.run([self.loss, self.train_step], feed_dict={self.X: X[j:j+param_batch_size], self.Yoh_: Yoh_[j:j+param_batch_size]})
            if i % param_print_step == 0:
                print("Iteration = ",i,", Loss = ",loss)

    """
        Arguments:
        - X: actual datapoints [NxD]
        Returns: predicted class probabilites [NxC]
    """
    def eval(self, X):
        return self.session.run(self.probs, feed_dict={self.X: X})

    def count_params(self):
        params_count = 0
        print("Counting params:")
        for v in tf.trainable_variables():
            print(v.name)
            tmp = 1
            for i in v.shape:
                tmp *= i
            params_count += tmp
        print("Params count = ",params_count)



if __name__ == "__main__":
    # initialize the random number generator
    np.random.seed(100)
    tf.set_random_seed(100)

    # instantiate the data X and the labels Yoh_
    X, Y_ = data.sample_gmm_2d(5, 3, 40)
    Yoh_ = data.class_to_one_hot(Y_)

    # build the graph:
    tfdeep = TFDeep([2, 10, 3], param_delta=0.01)
    tfdeep.count_params()

    # perform the training with given hyper-parameters:
    tfdeep.train(X, Yoh_, 100000)

    # predict probabilities of the data points
    probs = tfdeep.eval(X)
    Y = probs.argmax(axis=1)

    # print performance (per-class precision and recall)
    accuracy = data.eval_perf_multi(Y, Y_)

    print("Accuracy = ",accuracy)

    # draw results, decision surface
    decfun = lambda x: tfdeep.eval(x).max(axis=1)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)
    data.graph_data(X, Y_, Y, special=[])
    plt.show()
