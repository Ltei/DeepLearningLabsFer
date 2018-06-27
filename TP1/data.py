
import numpy as np;
import matplotlib.pyplot as plt


#
# Utils
#

def relu(x):
    return np.maximum(0.0, x)

def relu_deriv(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv(x):
    tmp = sigmoid(x)
    return tmp * (1.0 - tmp)


def class_to_one_hot(x):
    output = np.zeros((len(x), max(x)+1))
    output[range(len(x)), x] = 1
    return output

def log_likehood_loss(prob, y):
    return -y * np.log(prob) - (1 - y) * np.log(1 - prob)



#
# TP0 - Part 1
#

'''
    2D value generator with Gaussian distribution
'''
class Random2DGaussian:

    minx = 0
    maxx = 10
    miny = 0
    maxy = 10
    scale = 5

    def __init__(self):
        self.mean = [self.minx + np.random.random_sample()*(self.maxx-self.minx), self.miny + np.random.random_sample()*(self.maxy-self.miny)]

        D = [(np.random.random_sample() * (self.maxx-self.minx)/self.scale) ** 2, (np.random.random_sample() * (self.maxy-self.miny)/self.scale) ** 2]

        angle = np.random.random_sample() * 2 * np.pi
        cos = np.cos(angle)
        sin = np.sin(angle)
        R = [[cos, -sin], [sin, cos]]

        self.covarianceM = np.dot(np.dot(np.transpose(R), np.diag(D)), R)

    '''
        Generate n 2D values
    '''
    def get_sample(self, n):
        return np.random.multivariate_normal(self.mean, self.covarianceM, n)

def __test_random2dGaussian():
    G = Random2DGaussian()
    X = G.get_sample(100)
    print(X);
    plt.scatter(X[:,0], X[:,1])
    plt.show()



#
# TP0 - Part 2
#

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def cross_entropy_loss(prob, y):
    return -y * np.log(prob) - (1 - y) * np.log(1 - prob)
#def softmax(x):
#    exp_x_shifted = np.exp(x - np.max(x))
#    probs = exp_x_shifted / np.sum(exp_x_shifted)
#    return probs


'''
    Generate C samples of N random 2D values
    Return them and an array containing the labels for the C samples
'''
def sample_gauss_2d(C, N):
    X = []
    y = []
    for i in range(C):
        X.extend(Random2DGaussian().get_sample(N))
        y.extend([i]*N)
    return np.array(X), np.array(y)


def eval_perf_binary(Y, Y_):
  tp = sum(np.logical_and(Y==Y_, Y_==True))
  fn = sum(np.logical_and(Y!=Y_, Y_==True))
  tn = sum(np.logical_and(Y==Y_, Y_==False))
  fp = sum(np.logical_and(Y!=Y_, Y_==False))
  recall = tp / (tp + fn)
  precision = tp / (tp + fp)
  accuracy = (tp + tn) / (tp+fn + tn+fp)
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

def eval_perf_multi(Y, Y_):
    # tmp = np.argmax(Y, axis=1) == np.argmax(Y_, axis=1)
    # tmp = np.sum(tmp) / Y.shape[0]
    # return tmp
  pr = []
  n = max(Y_) + 1
  M = np.bincount(n * Y_ + Y, minlength=n*n).reshape(n, n)
  for i in range(n):
    tp_i = M[i,i]
    fn_i = np.sum(M[i,:]) - tp_i
    fp_i = np.sum(M[:,i]) - tp_i
    tn_i = np.sum(M) - fp_i - fn_i - tp_i
    recall_i = tp_i / (tp_i + fn_i)
    precision_i = tp_i / (tp_i + fp_i)
    pr.append( (recall_i, precision_i) )

  accuracy = np.trace(M)/np.sum(M)

  return accuracy



#
# TP0 - Part 3
#

"""
    Creates a surface plot (visualize with plt.show)
    Arguments:
    function: surface to be plotted
    rect:     function domain provided as:
                ([x_min,y_min], [x_max,y_max])
    offset:   the level plotted as a contour plot
    Returns:
    None
"""
def graph_surface(function, rect, offset=0.5, width=256, height=256):

  lsw = np.linspace(rect[0][1], rect[1][1], width)
  lsh = np.linspace(rect[0][0], rect[1][0], height)
  xx0,xx1 = np.meshgrid(lsh, lsw)
  grid = np.stack((xx0.flatten(),xx1.flatten()), axis=1)
  #get the values and reshape them
  values=function(grid).reshape((width,height))
  # fix the range and offset
  delta = offset if offset else 0
  maxval=max(np.max(values)-delta, - (np.min(values)-delta))
  # draw the surface and the offset
  plt.pcolormesh(xx0, xx1, values, vmin=delta-maxval, vmax=delta+maxval, cmap='jet')
  if offset != None:
    plt.contour(xx0, xx1, values, colors='black', levels=[offset])

"""
    Creates a scatter plot (visualize with plt.show)
    Arguments:
        X:       datapoints
        Y_:      groundtruth classification indices
        Y:       predicted class indices
        special: use this to emphasize some points
    Returns:
        None
"""
def graph_data(X,Y_, Y, special=[]):
  # colors of the datapoint markers
  palette=([0.5,0.5,0.5], [1,1,1], [0.2,0.2,0.2])
  colors = np.tile([0.0,0.0,0.0], (Y_.shape[0],1))
  for i in range(len(palette)):
    colors[Y_==i] = palette[i]
  # sizes of the datapoint markers
  sizes = np.repeat(20, len(Y_))
  sizes[special] = 40
  # draw the correctly classified datapoints
  good = (Y_==Y)
  plt.scatter(X[good,0],X[good,1], c=colors[good], s=sizes[good], marker='o')
  # draw the incorrectly classified datapoints
  bad = (Y_!=Y)
  plt.scatter(X[bad,0],X[bad,1], c=colors[bad], s=sizes[bad], marker='s')

def myDummyDecision(X):
    scores = X[:,0] + X[:,1] - 5
    return scores

def __test_graph_surface_data():
    # get the training dataset
    X,Y_ = sample_gauss_2d(2, 100)
    # get the class predictions
    Y = myDummyDecision(X)>0.5
    # graph the data points
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    print("Bbox0 = ",bbox)
    graph_surface(myDummyDecision, bbox, offset=0)
    graph_data(X, Y_, Y)
    # show the results
    plt.show()



#
# TP0 - Part 5
#



#
# TP1 - Part 1
#

'''
    Returns X: data in a matrix [K*N x 2 ], Y_: class indices of data [K*N]
'''
def sample_gmm_2d(K, C, N):
    X = []
    Y_ = []
    for i in range(K):
        X.append(Random2DGaussian().get_sample(N))
        Y_.append([np.random.randint(C)] * N)
    return np.vstack(X), np.hstack(Y_)

def __test_sample_gmm_2d():
    X, Y_ = sample_gmm_2d(4, 2, 30)
    Y = myDummyDecision(X) > 0.5
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    print("Bbox1 = ",bbox)
    np.linspace(bbox[0][1], bbox[1][1], 256)
    graph_surface(myDummyDecision, bbox, offset=0)
    graph_data(X, Y_, myDummyDecision(X)>0.5)



def sample_gmm(ncomponents, nclasses, nsamples):
  # create the distributions and groundtruth labels
  Gs=[]
  Ys=[]
  for i in range(ncomponents):
    Gs.append(Random2DGaussian())
    Ys.append(np.random.randint(nclasses))
  # sample the dataset
  X = np.vstack([G.get_sample(nsamples) for G in Gs])
  Y_= np.hstack([[Y]*nsamples for Y in Ys])

  return X,Y_


if __name__=="__main__":
    np.random.seed(100)
    __test_random2dGaussian()
    #__test_graph_surface_data()
    #__test_sample_gmm_2d()
