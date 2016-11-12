
import theano
import theano.tensor as T
import numpy as np

from lasagne.layers import InputLayer, DenseLayer, ElemwiseSumLayer, ConcatLayer
from lasagne.layers import LSTMLayer, EmbeddingLayer, ReshapeLayer
from lasagne.layers import get_output, get_all_params
from lasagne.layers import get_all_param_values, set_all_param_values
from lasagne.nonlinearities import softmax, tanh
import lasagne


def bilstm_layer(input_layer, lstm_dim, batch_size, context,
                 depth=1, grad_clip=100):
    """
    batch_size: int or symbolic_var (input_var.shape[0])
    context: int
    """
    lstm = emb
    for n in range(n_layers):
        fwd_lstm = LSTMLayer(lstm, lstm_dim, grad_clipping=grad_clip)
        bwd_lstm = LSTMLayer(lstm, lstm_dim, backwards=True,
                             grad_clipping=grad_clip)
        # No need to reverse output of bwd_lstm since backwards is defined:
        # backwards : bool
        #   process the sequence backwards and then reverse the output again
        #   such that the output from the layer is always from x1x1 to xnxn.
        # concat over lstm output dim (axis=2)
        lstm = ConcatLayer(incomings=[fwd_lstm, bwd_lstm], axis=2)
        # reshape for dense
        lstm = ReshapeLayer(lstm, (-1, lstm_dim * 2))
        lstm = DenseLayer(lstm, num_units=lstm_dim, nonlinearity=tanh)
        # reshape back to original input
        lstm = ReshapeLayer(lstm, (batch_size, context * 2, lstm_dim))
    return lstm


class BiLSTM(object):
    def __init__(self, emb_dim, lstm_dim, vocab_size, context,
                 depth=1, grad_clip=100):
        self.emb_dim = emb_dim
        self.lstm_dim = lstm_dim
        self.vocab_size = vocab_size

        # Input is integer matrices (batch_size, seq_length)
        input_layer = InputLayer(shape=(None, None),
                                 input_var=T.imatrix())
        self.emb_W = np.random.uniform(size=(vocab_size, emb_dim)) \
                              .astype(np.float32)
        emb = EmbeddingLayer(input_layer, input_size=vocab_size,
                             output_size=emb_dim, W=self.emb_W)
        batch_size, _ = input_layer.input_var.shape
        lstm = bilstm_layer(emb, lstm_dim, batch_size, context,
                            depth=depth, grad_clip=grad_clip)
        self.output = DenseLayer(
            lstm, num_units=vocab_size, nonlinearity=softmax)

        # T.nnet.categorical_crossentropy allows to represent true distribution
        # as an integer vector (implicitely casting to a one-hot matrix)
        lr, targets = T.fscalar('lr'), T.ivector('targets')
        pred = get_output(self.output)
        loss = T.nnet.categorical_crossentropy(pred, targets).mean()
        acc = T.mean(T.eq(T.argmax(pred, axis=1), targets),
                     dtype=theano.config.floatX)
        params = get_all_params(self.output, trainable=True)
        updates = lasagne.updates.rmsprop(loss, params, lr)

        print("Compiling training function")
        self._train = theano.function(
            [input_layer.input_var, targets, lr],
            [loss, acc],
            updates=updates,
            allow_input_downcast=True)

        test_pred = get_output(self.output, deterministic=True)
        test_loss = T.nnet.categorical_crossentropy(test_pred, targets).mean()
        test_acc = T.mean(T.eq(T.argmax(test_pred, axis=1), targets),
                          dtype=theano.config.floatX)

        print("Compiling test function")
        self._test = theano.function(
            [input_layer.input_var, targets],
            [test_loss, test_acc],
            allow_input_downcast=True)

        print("Compiling predict function")
        self._predict = theano.function(
            [input_layer.input_var],
            test_pred,
            allow_input_downcast=True)

    def train_on_batch(self, batch_X, batch_y, lr=0.01, shuffle=True):
        """
        Parameters:
        -----------
        batch_X: np.array(size=(batch_size, seq_len), dtype=np.int)
        batch_y: np.array(size=(batch_size, vocab_size))
        """
        assert batch_X.shape[0] == batch_y.shape[0]
        if shuffle:
            p = np.random.permutation(batch_X.shape[0])
            return self._train(batch_X[p, :], batch_y[p], lr)
        else:
            return self._train(batch_X, batch_y, lr)

    def test_on_batch(self, batch_X, batch_y, **kwargs):
        return self._test(batch_X, batch_y)

    def fit(self, batch_gen, epochs, batch_size, batches=1, **kwargs):
        """
        Parameters:
        -----------
        batch_gen: generator function returning batch tuples (X, y)
        epochs: int, number of epochs
        batch_size: int, input to batch_gen
        batches: int, how many batches in between loss report
        """
        for e in range(epochs):
            losses, accs = [], []
            for b, (X, y) in enumerate(batch_gen(batch_size)):
                loss, acc = self.train_on_batch(X, y, **kwargs)
                losses.append(loss), accs.append(acc)
                if b % batches == 0:
                    yield False, e, b, losses, accs
            yield True, e, b, losses, accs

    def predict(self, X, max_n=1, return_probs=False):
        out = self._predict(X)
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
    parser.add_argument('-n', '--num_batches', type=int, default=10000)
    parser.add_argument('-r', '--root', type=str, required=True)
    parser.add_argument('-p', '--model_prefix', type=str, required=True)
    parser.add_argument('-d', '--db', type=str, default='db.json')
    parser.add_argument('-L', '--loss', type=int, default=10,
                        help='report loss every l batches')

    from casket.nlp_utils import Corpus, Indexer
    from casket import Experiment as E
    import os

    args = parser.parse_args()
    root = args.root
    path = args.db
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
            batch_size=batch_size, indexer=idxr, concat=True, mode='chars')
        for idx, (X, y) in enumerate(gen):
            if idx <= NUM_BATCHES:
                yield (np.asarray(X), np.asarray(y))

    test_X, test_y = next(batch_gen(2000, corpus=test))
    dev_X, dev_y = next(batch_gen(1000, corpus=dev))

    vocab_size = idxr.vocab_len()
    bilstm = BiLSTM(emb_dim=EMB_DIM, lstm_dim=LSTM_DIM,
                    vocab_size=vocab_size, context=CONTEXT, depth=RNN_LAYERS)
    
    print("Starting training")
    db = E.use(path, exp_id='lasagne-bilstm').model("")
    with db.session(vars(args), ensure_unique=False) as session:
        for flag, epoch, batch, losses, accs in bilstm.fit(
                batch_gen, EPOCHS, BATCH_SIZE, batches=LOSS):
            if flag:                # do epoch testing
                loss, acc = bilstm.test_on_batch(dev_X, dev_y)
                session.add_epoch(epoch, {'loss': float(loss),
                                          'acc': float(acc)})
                print("Epoch test accuracy [%f]" % acc)
            else:                   # do batch logging
                print("Epoch [%d], batch [%d], Avg. loss [%f], Acc [%f]" %
                      (epoch, batch, np.mean(losses), np.mean(accs)), end='\r')
        loss, acc = bilstm.test_on_batch(test_X, test_y)
        session.add_result({'test_acc': float(acc), 'test_loss': float(loss)})

    model.save(args.model_prefix + ".weights")
    idxr.save(args.model_prefix + '_indexer.json')
