
import numpy as np;
import matplotlib.pyplot as plt
import data



#
# Part 2
#

'''
Arguments
    X:  data, np.array NxD
    Y_: class indices, np.array Nx1

Return values
    w, b: parameters of binary logistic regression
'''
def binlogreg_train(X,Y_, param_niter = 100000, param_delta = 0.1):
    nb_samples = len(X)
    w = np.random.randn(2)
    b = 0
    for i in range(param_niter):
        scores = np.dot(X, w) + b
        probs = data.sigmoid(scores)
        loss  = np.sum(data.cross_entropy_loss(probs, Y_))

        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # derivative of the loss funciton with respect to classification scores
        dL_dscores = probs - Y_ #(1 / len(Y_)) * (probs - Y_) * X  # N x 1

        # gradients with respect to parameters
        grad_w = 1.0/nb_samples  * np.dot(dL_dscores, X)     # D x 1
        grad_b = 1.0/nb_samples  * np.sum(dL_dscores)        # 1 x 1

        # modifying the parameters
        w += -param_delta * grad_w
        b += -param_delta * grad_b

    return w,b


'''
Arguments
    X:    data, np.array NxD
    w, b: logistic regression parameters

Return values
    probs: a posteriori probabilities for c1, dimensions Nx1
'''
def binlogreg_classify(X,w,b):
    return data.sigmoid(np.dot(X, w) + b)


np.random.seed(100)
X,Y_ = data.sample_gauss_2d(2, 100)
w,b = binlogreg_train(X, Y_, param_niter=0)
probs = binlogreg_classify(X, w,b)

Y = []
for prob in probs:
    if prob < 0.5:
        Y.append(False)
    else:
        Y.append(True)

accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
AP = data.eval_AP(Y_[probs.argsort()])
#print (accuracy, recall, precision, AP)
