
import numpy as np
import matplotlib.pyplot as plt
import data


#
# TP1 - Task 2
#

def fcann2_train(X, Y_, param_niter=100000, param_delta=0.05, param_lambda=1e-3, param_hidden_len=5, print_every=1000):
    input_count, input_len = X.shape
    Y_ = data.class_to_one_hot(Y_)
    output_len = len(Y_[0])

    W_1 = 2.0 * np.random.randn(input_len, param_hidden_len) - 1.0
    b_1 = 2.0 * np.random.randn(param_hidden_len) - 1.0
    W_2 = 2.0 * np.random.randn(param_hidden_len, output_len) - 1.0
    b_2 = 2.0 * np.random.randn(output_len) - 1.0

    print('--',X.shape)
    print(Y_.shape)
    print(W_1.shape)
    print(b_1.shape)
    print(W_2.shape)
    print(b_2.shape,'--')

    for i in range(param_niter):
        hidden = data.relu(np.dot(X, W_1) + b_1)
        output = data.sigmoid(np.dot(hidden, W_2) + b_2)
        loss = np.sum(data.log_likehood_loss(output, Y_)) / input_count

        if i % print_every == 0:
            print("Iteration = ",i,", Loss = ",loss)

        signal = output - Y_
        signal /= input_count

        grad_W2 = np.dot(hidden.T, signal)
        grad_b2 = np.sum(signal, axis=0, keepdims=True)

        grad_hidden = np.dot(signal, W_2.T)
        grad_hidden[hidden <= 0] = 0

        grad_W1 = np.dot(X.T, grad_hidden)
        grad_b1 = np.sum(grad_hidden, axis=0, keepdims=True)

        grad_W2 += param_lambda * W_2
        grad_W1 += param_lambda * W_1

        W_1 -= param_delta * grad_W1
        b_1 -= param_delta * grad_b1.reshape(param_hidden_len,)
        W_2 -= param_delta * grad_W2
        b_2 -= param_delta * grad_b2.reshape(output_len,)

    return W_1, b_1, W_2, b_2

def fcann2_classify(X, W_1, b_1, W_2, b_2):
    hidden = data.relu(np.dot(X, W_1) + b_1)
    return data.sigmoid(np.dot(hidden, W_2) + b_2)

def fcann2_classify_new(W_1, b_1, W_2, b_2):
    def classify(X):
        return fcann2_classify(X, W_1, b_1, W_2, b_2)[:,0]
    return classify

def __test_tp1_task2():
    np.random.seed(50)
    X, Y_ = data.sample_gmm(6, 2, 10)
    W1, b1, W2, b2 = fcann2_train(X, Y_, param_hidden_len=100)
    decfunc = fcann2_classify_new(W1, b1, W2, b2)

    Y = np.argmax(fcann2_classify(X, W1, b1, W2, b2), axis=1)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfunc, bbox, offset=0.5)
    data.graph_data(X, Y_, Y)

    plt.show()



if __name__=="__main__":
    __test_tp1_task2()
    #__test_tp1_task3()
