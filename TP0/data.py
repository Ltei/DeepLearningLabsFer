
import numpy as np;
import matplotlib.pyplot as plt


#
# Utils
#

def rescale_from_01(x, min, max):
    return min + x * (max - min)


#
# Part 1
#

np.random.seed(100);

class Random2DGaussian:

    minx = 0
    maxx = 10
    miny = 0
    maxy = 10
    scale = 5

    def __init__(self):
        self.mean = [rescale_from_01(np.random.random_sample(), self.minx, self.maxx), rescale_from_01(np.random.random_sample(), self.miny, self.maxy)]

        D = [(np.random.random_sample() * (self.maxx-self.minx)/self.scale) ** 2, (np.random.random_sample() * (self.maxy-self.miny)/self.scale) ** 2]

        angle = np.random.random_sample() * 2 * np.pi
        cos = np.cos(angle)
        sin = np.sin(angle)
        R = [[cos, -sin], [sin, cos]]

        self.covarianceM = np.dot(np.dot(np.transpose(R), np.diag(D)), R)

    def get_sample(self, n):
        return np.random.multivariate_normal(self.mean, self.covarianceM, n)


def test_part1():
    np.random.seed(10)
    G = Random2DGaussian()
    X = G.get_sample(100)
    print(X);
    plt.scatter(X[:,0], X[:,1])
    plt.show()



#
# Part 2
#

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def cross_entropy_loss(prob, y):
    return -y * np.log(prob) - (1 - y) * np.log(1 - prob)
#def softmax(x):
#    exp_x_shifted = np.exp(x - np.max(x))
#    probs = exp_x_shifted / np.sum(exp_x_shifted)
#    return probs


def sample_gauss_2d(C, N):
    X = []
    y = []
    for i in range(C):
        X.extend(Random2DGaussian().get_sample(N))
        y.extend([i]*N)
    return np.array(X), np.array(y)

def eval_perf_binary(Y, Y_):
  sum_true_pos = sum(np.logical_and(Y==Y_, Y_==True))
  sum_true_neg = sum(np.logical_and(Y==Y_, Y_==False))
  sum_false_pos = sum(np.logical_and(Y!=Y_, Y_==False))
  sum_false_neg = sum(np.logical_and(Y!=Y_, Y_==True))
  recall = sum_true_pos / (sum_true_pos + sum_false_neg)
  precision = sum_true_pos / (sum_true_pos + sum_false_pos)
  accuracy = (sum_true_pos + sum_true_neg) / (sum_true_pos+sum_false_neg + sum_true_neg + sum_false_pos)
  return accuracy, recall, precision

def eval_AP(ranked_labels):
  n = len(ranked_labels)
  pos = sum(ranked_labels)
  neg = n - pos

  tp = pos
  tn = 0
  fn = 0
  fp = neg

  sumprec=0
  for x in ranked_labels:
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    if x:
      sumprec += precision

    tp -= x
    fn += x
    fp -= not x
    tn += not x

  return sumprec/pos



#
# Part 3
#

def graph_data(X, Y_, Y):
    '''
    X  ... data (np. array Nx2)
    Y_ ... true classes (np.array Nx1)
    Y  ... predicted classes (np.array Nx1)
    '''
    size = 30
    class0Color = [0.4,0.4,0.4]
    class1Color = [0.8,0.8,0.8]

    colors = np.tile([0.0,0.0,0.0], (Y_.shape[0],1))
    colors[Y_==0] = class0Color
    colors[Y_==1] = class1Color

    good = (Y_==Y)
    plt.scatter(X[good,0],X[good,1], c=colors[good], s=size, marker='o')

    bad = (Y_!=Y)
    plt.scatter(X[bad,0],X[bad,1], c=colors[bad], s=size, marker='s')



#
# Part 4
#

def graph_surface(fun, rect, offset=0.5, width=256, height=256):
    '''
    fun    ... the decision function (Nx2)->(Nx1)
    rect   ... he domain in which we plot the data:
                ([x_min,y_min], [x_max,y_max])
    offset ... the value of the decision function on the border between the classes;
                we typically have:
                offset = 0.5 for probabilistic models
                    (e.g. logistic regression)
                offset = 0 for models which do not squash
                    classification scores (e.g. SVM)
    width,height ... rezolucija koordinatne mreže
    '''
    positionsX = np.linspace(rect[0][1], rect[1][1], width)
    positionsY = np.linspace(rect[0][0], rect[1][0], height)
    grid0,grid1 = np.meshgrid(positionsX, positionsY)
    grid = np.stack((grid0.flatten(), grid1.flatten()), axis=1)

    #get the values and reshape them
    values = fun(grid).reshape((width,height))

    # fix the range and offset
    delta = offset if offset else 0
    maxval=max(np.max(values)-delta, - (np.min(values)-delta))

    # draw the surface and the offset
    plt.pcolormesh(grid0, grid1, values, vmin=delta-maxval, vmax=delta+maxval)

    if offset != None:
        plt.contour(grid0, grid1, values, colors='black', levels=[offset])


def myDummyDecision(X):
    scores = X[:,0] + X[:,1] - 5
    return scores
def test_part4():
    # get the training dataset
    X,Y_ = sample_gauss_2d(2, 100)
    # get the class predictions
    Y = myDummyDecision(X)>0.5
    # graph the data points
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(myDummyDecision, bbox, offset=0)
    graph_data(X, Y_, Y)
    # show the results
    plt.show()

if __name__=="__main__":
    #test_part1()
    test_part4()


#
# Part 5
#
