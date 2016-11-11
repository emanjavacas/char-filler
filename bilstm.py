
import theano
import theano.tensor as T
import numpy as np

from lasagne.layers import InputLayer, DenseLayer, ElemwiseSumLayer
from lasagne.layers import LSTMLayer, EmbeddingLayer
from lasagne.layers import get_output, get_all_params
from lasagne.layers import get_all_param_values, set_all_param_values
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
        self.emb_W = np.random.uniform(size=(vocab_size, emb_dim))\
                              .astype(np.float32)
        emb_left = EmbeddingLayer(
            left_in, input_size=vocab_size, output_size=emb_dim, W=self.emb_W)
        emb_right = EmbeddingLayer(
            right_in, input_size=vocab_size, output_size=emb_dim, W=self.emb_W)
        merged = bilstm_layer(emb_left, emb_right, lstm_dim, n_layers=n_layers)
        self.output = DenseLayer(
            merged, num_units=vocab_size, nonlinearity=softmax)

        # T.nnet.categorical_crossentropy allows to represent true distribution
        # as an integer vector (implicitely casting to a one-hot matrix)
        lr, targets = T.fscalar('lr'), T.ivector('targets')
        network_output = get_output(self.output)
        cost = T.nnet.categorical_crossentropy(network_output, targets).mean()
        params = get_all_params(self.output, trainable=True)
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
            return self._train(
                batch_left[p, :], batch_right[p, :], batch_y[p], lr)
        else:
            return self._train(batch_left, batch_right, batch_y, lr)

    def test_on_batch(self, batch_left, batch_right, batch_y, **kwargs):
        pred = self.predict(batch_left, batch_right, max_n=False)
        return lasagne.objectives.categorical_accuracy(pred, batch_y, **kwargs)

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
                    yield False, e, b, losses
            yield True, e, b, losses

    def predict(self, left_in, right_in, max_n=1, return_probs=False):
        """
        probably needs embedding in a batch matrix of length 1 if
        left_in and right_in are integer vectors.
        """
        out = self._predict(left_in, right_in)
        idx = np.argsort(out)
        max_n = max_n or idx.shape[0]
        if return_probs:
            return idx[:max_n], out[idx[:max_n]]
        else:
            return idx[:max_n]

    def save(self, prefix):
        import json
        network_weights = get_all_param_values(self.output)
        np.savez(prefix + ".npz", network_weights)
        with open(prefix + ".json", "w") as f:
            f.write(json.dumps(
                {"vocab_size": self.vocab_size,
                 "emb_dim": self.emb_dim,
                 "lstm_dim": self.lstm_dim}))

    @classmethod
    def load(cls, prefix):
        import json
        with open(prefix + ".json") as data:
            pms = json.load(data)
        network = cls(pms['emb_dim'], pms['lstm_dim'], pms['vocab_size'])
        with np.load(prefix + '.npz') as data:
            set_all_param_values(network.output)
        return network


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-R', '--rnn_layers', type=int, default=1)
    parser.add_argument('-m', '--emb_dim', type=int, default=28)
    parser.add_argument('-l', '--lstm_dim', type=int, default=64)
    parser.add_argument('-D', '--dropout', type=float, default=0.0)
    parser.add_argument('-c', '--context', type=float, default=15)
    parser.add_argument('-b', '--batch_size', type=int, default=1024)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-n', '--num_batches', type=int, default=1000)
    parser.add_argument('-r', '--root', type=str, required=True)
    parser.add_argument('-p', '--model_prefix', type=str, required=True)
    parser.add_argument('-d', '--db', type=str, default='db.json')
    parser.add_argument('-L', '--loss', type=int, default=1,
                        help='report loss every l batches')

    from casket.nlp_utils import Corpus, Indexer
    import os

    args = parser.parse_args()
    root = args.root
    assert os.path.isdir(root), "Root path doesn't exist"

    BATCH_SIZE = args.batch_size
    NUM_BATCHES = args.num_batches
    EPOCHS = args.epochs
    RNN_LAYERS = args.rnn_layers
    EMB_DIM = args.emb_dim
    LSTM_DIM = args.lstm_dim
    DROPOUT = args.dropout
    CONTEXT = args.context
    LOSS = args.loss

    train = Corpus(root + "/train", context=CONTEXT)
    test = Corpus(root + "/test", context=CONTEXT)
    dev = Corpus(root + "/dev", context=CONTEXT)
    idxr = Indexer()
    idxr.fit(train.chars())

    def batch_gen(batch_size, corpus=train):
        gen = corpus.generate_batches(
            batch_size=batch_size, indexer=idxr, concat=False, mode='chars')
        for idx, (X, y) in enumerate(gen):
            if idx <= NUM_BATCHES:
                left, right = list(zip(*X))
                yield (np.asarray(left), np.asarray(right)), np.asarray(y)

    test_X_l, test_X_r, test_y = next(batch_gen(2000, corpus=test))
    dev_X_l, dev_X_r, dev_y = next(batch_gen(2000, corpus=dev))

    vocab_size = idxr.vocab_len()
    bilstm = BiLSTM(emb_dim=EMB_DIM, lstm_dim=LSTM_DIM, vocab_size=vocab_size)
    print("Starting training")
    for flag, epoch, batch, losses in bilstm.fit(
            batch_gen, EPOCHS, BATCH_SIZE, batches=LOSS):
        if flag:                # do epoch testing
            acc = bilstm.test_on_batch(test_X_l, text_X_r, test_y)
            print("Epoch test accuracy [%f]" % acc)
        else:                   # do batch logging
            acc = bilstm.test_on_batch(dev_X_l, dev_X_r, dev_y)
            print("Epoch [%d], batch [%d], Avg. loss [%f], Acc [%f]" %
                  (epoch, batch, np.mean(losses), acc), end='\r')
