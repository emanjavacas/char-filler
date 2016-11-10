
import theano
import theano.tensor as T
import numpy as np

from lasagne.layers import InputLayer, DenseLayer, ElemwiseSumLayer
from lasagne.layers import LSTMLayer, EmbeddingLayer
from lasagne.layers import get_output, get_all_params
from lasagne.nonlinearities import softmax, tanh
import lasagne


def bilstm_layer(input_left, input_right, lstm_dim, n_layers=1):
    for n in range(n_layers):
        left_lstm = LSTMLayer(
            input_left, lstm_dim, only_return_final=True, backwards=True)
        right_lstm = LSTMLayer(
            input_right, lstm_dim, only_return_final=True, backwards=False)
        dense_left = DenseLayer(
            left_lstm, num_units=lstm_dim, nonlinearity=tanh)
        dense_right = DenseLayer(
            right_lstm, num_units=lstm_dim, nonlinearity=tanh)
        input_left = dense_left
        input_right = dense_right
    return ElemwiseSumLayer([dense_left, dense_right])


class BiLSTM(object):
    def __init__(self, emb_dim, lstm_dim, vocab_size, n_layers=1):
        self.emb_dim = emb_dim
        self.lstm_dim = lstm_dim
        self.vocab_size = vocab_size

        # Input is integer matrices (batch_size, seq_length)
        left_in = InputLayer(shape=(None, None), input_var=T.imatrix())
        right_in = InputLayer(shape=(None, None), input_var=T.imatrix())
        self.emb_W = np.random.uniform(size=(vocab_size, emb_dim)).astype(np.float32)
        emb_left = EmbeddingLayer(
            left_in, input_size=vocab_size, output_size=emb_dim, W=self.emb_W)
        emb_right = EmbeddingLayer(
            right_in, input_size=vocab_size, output_size=emb_dim, W=self.emb_W)
        merged = bilstm_layer(emb_left, emb_right, lstm_dim, n_layers=n_layers)
        output = DenseLayer(merged, num_units=vocab_size, nonlinearity=softmax)

        # T.nnet.categorical_crossentropy allows to represent true distribution
        # as an integer vector (implicitely casting to a one-hot matrix)
        lr, targets = T.fscalar('lr'), T.ivector('targets')
        network_output = get_output(output)
        cost = T.nnet.categorical_crossentropy(network_output, targets).mean()
        params = get_all_params(output, trainable=True)
        updates = lasagne.updates.adagrad(cost, params, lr)

        print("Compiling training function")
        self._train = theano.function(
            [left_in.input_var, right_in.input_var, targets, lr],
            cost,
            updates=updates,
            allow_input_downcast=True)

        print("Compiling predict function")
        self._predict = theano.function(
            [left_in.input_var, right_in.input_var],
            network_output,
            allow_input_downcast=True)

    def train_on_batch(self, batch_left, batch_right, batch_y,
                       lr=0.01, shuffle=False):
        """
        Parameters:
        -----------
        batch_left: np.array(size=(batch_size, left_seq_len), dtype=np.int)
        batch_left: np.array(size=(batch_size, right_seq_len), dtype=np.int)
        batch_y: np.array(size=(batch_size, vocab_size))
        """
        assert batch_left.shape[0] == batch_right.shape[0] == batch_y.shape[0]
        if shuffle:
            p = np.random.permutation(batch_left.shape[0])
            return self._train(batch_left[p, :], batch_right[p, :], batch_y[p], lr)
        else:
            return self._train(batch_left, batch_right, batch_y, lr)

    def fit(self, batch_gen, epochs, batch_size, batches=1, **kwargs):
        """
        Parameters:
        -----------
        batch_gen: generator function returning batch tuples ((left, right), y)
        epochs: int, number of epochs
        batch_size: int, input to batch_gen
        batches: int, how many batches in between loss report
        """
        for e in range(epochs):
            losses = []
            for b, ((left, right), y) in enumerate(batch_gen(batch_size)):
                losses.append(self.train_on_batch(left, right, y, **kwargs))
                if b % batches == 0:
                    print("Epoch [%d], batch [%d], Avg. training loss [%f]" %
                          (e, b, np.mean(losses)))
            yield losses

    def predict(self, left_in, right_in, max_n=False, return_probs=False):
        """
        probably needs embedding in a batch matrix of length 1 if
        left_in and right_in are integer vectors.
        """
        out = self._predict(left_in, right_in)
        if max_n:
            idx = np.argsort(out)
            if return_probs:
                return idx[:max_n], out[idx[:max_n]]
            else:
                return idx[:max_n]
        else:
            if return_probs:
                return np.argmax(out), out[np.argmax(out)]
            else:
                return np.argmax(out)

    def save(self):
        pass

    def load(self):
        pass


def batches(chrs, idxr=None, max_window_len=15):
    chrs = list(chrs)
    def aux(batch_size):
        batch_left, batch_right, batch_y = [], [], []
        for idx, c in enumerate(chrs[max_window_len: len(chrs)-max_window_len]):
            if len(batch_y) > 0 and len(batch_y) % batch_size == 0:
                yield (np.asarray(batch_left), np.asarray(batch_right)), \
                    np.asarray(batch_y)
                batch_left, batch_right, batch_y = [], [], []
            idx += max_window_len
            left = chrs[max(0, idx - max_window_len): idx]
            right = chrs[idx + 1: min(len(chrs) + 1, idx + 1 + max_window_len)]
            y = chrs[idx]
            if idxr is not None:
                left, right = idxr.transform(left), idxr.transform(right)
                y = idxr.encode(y)
            batch_left.append(left), batch_right.append(right), \
                batch_y.append(y)
    return aux


if __name__ == '__main__':
    from casket.nlp_utils import Corpus, Indexer
    from itertools import islice

    train = Corpus("/home/enrique/corpora/EEBO/train")
    idxr = Indexer()
    idxr.fit(islice(train.chars(), 100000))
    batch_gen = batches(islice(train.chars(), 100000), idxr=idxr)
    vocab_size = idxr.vocab_len()
    bilstm = BiLSTM(emb_dim=64, lstm_dim=128, vocab_size=vocab_size)
    for loss in bilstm.fit(batch_gen, 5, 1024, batches=100):
        print("Epoch loss:", np.mean(loss))
