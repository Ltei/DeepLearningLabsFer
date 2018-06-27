
import numpy as np
from dataset import Dataset
from sklearn.metrics import log_loss


UPDATE_MAGNITUDE = 5



def softmax(x):
    assert len(x.shape) == 3
    s = np.max(x, axis=2)
    s = s[:, :, np.newaxis]
    e_x = np.exp(x - s)
    div = np.sum(e_x, axis=2)
    div = div[:, :, np.newaxis]
    return e_x / div


def to_one_hot(x, vocab_size):
    result = np.zeros([*x.shape, vocab_size], dtype=np.float32)
    for idx in np.ndindex(*x.shape):
        result[idx][x[idx]] = 1
    return result

def debug_ndarray(name, array):
    print(name," : ",array.shape," ",np.min(array)," ",np.average(array)," ",np.max(array))


class Network:

    def __init__(self, hidden_size, sequence_length, vocab_size, learning_rate):
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate

        scale = 1.0 / np.sqrt(hidden_size)

        self.U = np.random.normal(size=[vocab_size, hidden_size], scale=1.0 / np.sqrt(hidden_size))  # ... input projection
        self.W = np.random.normal(size=[hidden_size, hidden_size], scale=1.0 / np.sqrt(hidden_size))  # ... hidden-to-hidden projection
        self.b = np.zeros([1, hidden_size])

        self.V = np.random.normal(size=[hidden_size, vocab_size], scale=1.0 / np.sqrt(vocab_size))  # ... output projection
        self.c = np.zeros([1, vocab_size])  # ... output bias

        # memory of past gradients - rolling sum of squares for Adagrad
        self.memory_U, self.memory_W, self.memory_V = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.V)
        self.memory_b, self.memory_c = np.zeros_like(self.b), np.zeros_like(self.c)


    # A single time step forward of a recurrent neural network with a
    # hyperbolic tangent nonlinearity.
    # x - input data (minibatch size x input dimension)
    # h_prev - previous hidden state (minibatch size x hidden size)
    # U - input projection matrix (input dimension x hidden size)
    # W - hidden to hidden projection matrix (hidden size x hidden size)
    # b - bias of shape (hidden size x 1)
    def rnn_step_forward(self, x, h_prev, U, W, b):
        batch_size = x.shape[0]
        assert x.shape == (batch_size, self.vocab_size)
        assert h_prev.shape == (batch_size, self.hidden_size)
        h_current = np.tanh(np.dot(h_prev, W) + np.dot(x, U) + b)
        cache = (x, h_prev, h_current)
        return h_current, cache


    # Full unroll forward of the recurrent neural network with a
    # hyperbolic tangent nonlinearity
    # x - input data for the whole time-series (minibatch size x sequence_length x input dimension)
    # h0 - initial hidden state (minibatch size x hidden size)
    # U - input projection matrix (input dimension x hidden size)
    # W - hidden to hidden projection matrix (hidden size x hidden size)
    # b - bias of shape (hidden size x 1)
    def rnn_forward(self, x, h0, U, W, b):
        batch_size = x.shape[0]
        assert x.shape == (batch_size, self.sequence_length, self.vocab_size)
        assert h0.shape == (batch_size, self.hidden_size)
        h = [h0]
        cache = []
        for i in range(x.shape[1]):
            tmp_h, tmp_cache = self.rnn_step_forward(x[:,i], h[-1], U, W, b)
            h.append(tmp_h)
            cache.append(tmp_cache)
        h = h[1:]
        assert len(h) == self.sequence_length
        assert h[0].shape == (batch_size, self.hidden_size)
        return np.array(h).transpose(1,0,2), cache


    # A single time step backward of a recurrent neural network with a
    # hyperbolic tangent nonlinearity.
    # grad_next - upstream gradient of the loss with respect to the next hidden state and current output
    # cache - cached information from the forward pass
    def rnn_step_backward(self, grad_next, cache):
        div = grad_next.shape[0]
        x, h_prev, h_current = cache
        grad_next = grad_next * (1 - np.square(h_current)) # tanh backward
        dh_prev = np.dot(grad_next, self.W.T)
        dU = np.dot(x.T, grad_next) / div
        dW = np.dot(h_prev.T, grad_next) / div
        db = np.sum(grad_next, axis=0) / div
        return dh_prev, dU, dW, np.reshape(db, [1, self.hidden_size])


    # Full unroll forward of the recurrent neural network with a
    # hyperbolic tangent nonlinearity
    # compute and return gradients with respect to each parameter
    # for the whole time series.
    # Why are we not computing the gradient with respect to inputs (x)?
    def rnn_backward(self, dh_list, cache_list):
        dh_prev, dU_sum, dW_sum, db_sum = self.rnn_step_backward(dh_list[:,-1], cache_list[-1])
        for i in reversed(range( len(cache_list) - 1 )):
            dh_prev, dU, dW, db = self.rnn_step_backward(dh_list[:,i] + dh_prev, cache_list[i])
            dU_sum += dU
            dW_sum += dW
            db_sum += db
        return dU_sum, dW_sum, db_sum


    # Calculate the output probabilities of the network
    def output(self, h, V, c, test=True):
        if test:
            batch_size = h.shape[0]
            assert h.shape == (batch_size, self.sequence_length, self.hidden_size)
            assert V.shape == (self.hidden_size, self.vocab_size)
            assert c.shape == (1, self.vocab_size)
        result = softmax(np.dot(h, V) + c)
        if test:
            assert result.shape == (batch_size, self.sequence_length, self.vocab_size,)
        return result


    # Calculate the loss of the network for each of the outputs
    # h - hidden states of the network for each timestep.     [batch_size x sequence_len x hidden_size]
    # V - the output projection matrix of dimension           [hidden_size x vocabulary_size]
    # c - the output bias of dimension                        [vocabulary_size]
    # y - the true class distribution - a tensor of dimension [batch_size x sequence_length x vocabulary_size]
    #     you need to do this conversion prior to
    #     passing the argument. A fast way to create a one-hot vector from
    #     an id could be something like the following code:
    #   y[batch_id][timestep] = np.zeros((vocabulary_size, 1))
    #   y[batch_id][timestep][batch_y[timestep]] = 1
    #     where y might be a list or a dictionary.
    def output_loss_and_grads(self, h, V, c, y):
        batch_size = y.shape[0]
        assert h.shape == (batch_size, self.sequence_length, self.hidden_size)
        assert V.shape == (self.hidden_size, self.vocab_size)
        assert c.shape == (1, self.vocab_size)
        assert y.shape == (batch_size, self.sequence_length, self.vocab_size)

        yhat = self.output(h, V, c)
        loss = log_loss(y.reshape(-1, self.vocab_size), yhat.reshape(-1, self.vocab_size)) * self.sequence_length
        d_loss = yhat - y
        dV = np.zeros_like(V)
        dc = np.zeros_like(c)
        dh = []

        for i in range(self.sequence_length):
            dV += np.dot(h[:,i].T, d_loss[:,i]) / batch_size
            dc += np.average(d_loss[:,i], axis=0)
            dh.append(np.dot(d_loss[:,i], V.T))

        return loss, np.array(dh).transpose(1, 0, 2), dV, dc


    # update memory matrices
    # perform the Adagrad update of parameters
    def update(self, dU, dW, db, dV, dc, epsilon=1e-6):
        assert dU.shape == self.memory_U.shape
        assert dW.shape == self.memory_W.shape
        assert db.shape == self.memory_b.shape
        assert dV.shape == self.memory_V.shape
        assert dc.shape == self.memory_c.shape

        self.memory_U += np.square(dU)
        self.memory_W += np.square(dW)
        self.memory_b += np.square(db)
        self.memory_V += np.square(dV)
        self.memory_c += np.square(dc)

        self.U -= self.learning_rate * dU / np.sqrt(self.memory_U + epsilon)
        self.W -= self.learning_rate * dW / np.sqrt(self.memory_W + epsilon)
        self.b -= self.learning_rate * db / np.sqrt(self.memory_b + epsilon)
        self.V -= self.learning_rate * dV / np.sqrt(self.memory_V + epsilon)
        self.c -= self.learning_rate * dc / np.sqrt(self.memory_c + epsilon)


    def step(self, h0, x, y):
        batch_size = x.shape[0]
        assert h0.shape == (batch_size, self.hidden_size)
        assert x.shape == (batch_size, self.sequence_length, self.vocab_size)
        assert y.shape == (batch_size, self.sequence_length, self.vocab_size)

        h, cache = self.rnn_forward(x, h0, self.U, self.W, self.b)
        loss, dh, dV, dc = self.output_loss_and_grads(h, self.V, self.c, y)
        dU, dW, db = self.rnn_backward(dh, cache)

        dU = np.clip(dU, -UPDATE_MAGNITUDE, UPDATE_MAGNITUDE)
        dW = np.clip(dW, -UPDATE_MAGNITUDE, UPDATE_MAGNITUDE)
        db = np.clip(db, -UPDATE_MAGNITUDE, UPDATE_MAGNITUDE)
        dV = np.clip(dV, -UPDATE_MAGNITUDE, UPDATE_MAGNITUDE)
        dc = np.clip(dc, -UPDATE_MAGNITUDE, UPDATE_MAGNITUDE)
        self.update(dU, dW, db, dV, dc)

        return loss, h[:, -1, :]

    def sample(self, seed, n_sample):
        seed_one_hot = to_one_hot(seed, self.vocab_size)

        h = np.zeros((1, self.hidden_size))
        for i in range(seed_one_hot.shape[0]):
            h, _ = self.rnn_step_forward(seed_one_hot[1, np.newaxis, :], h, self.U, self.W, self.b)

        sample = np.zeros((n_sample, ), dtype=np.int32)
        sample[:len(seed)] = seed
        for i in range(len(seed), n_sample):
            model_out = self.output(h[np.newaxis, :, :], self.V, self.c, test=False)
            sample[i] = np.random.choice(np.arange(model_out.shape[-1]), p=model_out.ravel())

            model_out[:] = 0
            model_out = model_out.reshape(1, -1)
            model_out[0, sample[i]] = 1
            h, _ = self.rnn_step_forward(model_out, h, self.U, self.W, self.b)

        return sample

def run_language_model(dataset, max_epochs, hidden_size=100, sequence_length=30, learning_rate=0.1, sample_every=500):
    np.random.seed(10)

    vocab_size = len(dataset.sorted_chars)
    RNN = Network(hidden_size, sequence_length, vocab_size, learning_rate)

    print("num batches = ",dataset.num_batches)
    h0 = np.zeros((dataset.batch_size, hidden_size))
    current_epoch = 0
    batch = 0
    average_loss = 0

    while current_epoch < max_epochs:
        e, x, y = dataset.next_minibatch()
        assert x.shape == (dataset.batch_size, sequence_length)
        assert y.shape == (dataset.batch_size, sequence_length)
        assert h0.shape == (dataset.batch_size, hidden_size)

        if e:
            current_epoch += 1
            h0 = np.zeros((dataset.batch_size, hidden_size))
            # why do we reset the hidden state here?
            # because we start a new sequence

        # One-hot transform the x and y batches
        x_oh = to_one_hot(x, vocab_size)
        y_oh = to_one_hot(y, vocab_size)

        # Run the recurrent network on the current batch
        # Since we are using windows of a short length of characters, the step function should return the hidden state at the end
        # of the unroll. You should then use that hidden state as the input for the next minibatch. In this way, we artificially
        # preserve context between batches.
        loss, h0 = RNN.step(h0, x_oh, y_oh)
        average_loss = 0.9 * average_loss + 0.1 * loss

        if batch % sample_every == 0:
            print("Epoch = ",current_epoch,"/",max_epochs,", Batch ",batch,"/",dataset.num_batches,", Loss = ",average_loss)
            sample = RNN.sample(dataset.encode("HAN:\nIs that good or bad?\n\n"), 200)
            print(''.join(dataset.decode(sample)))

        batch += 1

    h0, seed_onehot, sample = None, None, None
    # inicijalizirati h0 na vektor nula
    # seed string pretvoriti u one-hot reprezentaciju ulaza

    return sample


if __name__ == "__main__":
    np.random.seed(10)
    dataset = Dataset()
    dataset.preprocess("selected_conversations.txt")
    dataset.create_minibatches()
    run_language_model(dataset, 1000)
