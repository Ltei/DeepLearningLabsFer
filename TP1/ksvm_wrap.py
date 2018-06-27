
from sklearn import svm
import numpy as np
import data
import matplotlib.pyplot as plt


class KSVMWrap:

    '''
        Constructs the wrapper and trains the RBF SVM classifier
        X,Y_:            data and indices of correct data classes
        param_svm_c:     relative contribution of the data cost
        param_svm_gamma:  RBF kernel width
    '''
    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        self.svc = svm.SVC(C = param_svm_c, gamma = param_svm_gamma);
        self.svc.fit(X, Y_)

    '''
        Predicts and returns the class indices of data X
    '''
    def predict(self, X):
        return self.svc.predict(X)

    '''
        Returns the classification scores of the data
        (you will need this to calculate average precision).
    '''
    def get_scores(self, X):
        return self.svc.decision_function(X)

    '''
        Indices of data chosen as support vectors
    '''
    def support(self):
        return self.svc.support_


def __test():
    np.random.seed(100)
    X, Y_ = data.sample_gmm(6, 2, 10)
    svm = KSVMWrap(X, Y_)
    Y = svm.predict(X)

    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    print("Accuracy : ", accuracy)
    print("Precision : ", precision)
    print("Recall : ", recall)

    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(svm.get_scores, bbox, offset=0)
    data.graph_data(X, Y_, Y, special=svm.support())
    plt.show()


if __name__ == "main" :
    __test()
