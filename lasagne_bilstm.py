
import theano
import theano.tensor as T
import numpy as np

from lasagne.layers import InputLayer, DenseLayer, ElemwiseSumLayer
from lasagne.layers import LSTMLayer, GRULayer
from lasagne.layers import EmbeddingLayer, ReshapeLayer, dropout, flatten
from lasagne.layers import get_output, get_all_params
from lasagne.layers import get_all_param_values, set_all_param_values
from lasagne.nonlinearities import softmax, tanh
import lasagne

np.random.seed(1002)

# Global model defaults
defs = {
    'depth': 1,
    'cell': 'lstm',
    'emb_dim': 64,
    'rnn_dim': 124,
    'hid_dim': 264,
    'dropout': 0.0,
    'context': 15,
    'add_dense': False
}


def accuracy(pred, target):
    return T.mean(T.eq(T.argmax(pred, axis=1), target),
                  dtype=theano.config.floatX)


def bid_layer(input_layer, rnn_dim, batch_size, rnn_shape, cell,
              add_dense=True, dropout_p=0.2, depth=1, **cell_args):
    """
    batch_size: int or symbolic_var (e.g. input_var.shape[0])
    context: int
    """
    if cell == 'lstm':
        cell = LSTMLayer
    elif cell == 'gru':
        cell = GRULayer
    else:
        raise ValueError('cell must be one of "lstm", "gru"')
    rnn = input_layer
    for n in range(depth):
        fwd = cell(rnn, rnn_dim, only_return_final=False, **cell_args)
        # No need to reverse output of bwd_lstm since backwards is defined:
        # backwards : bool
        #   process the sequence backwards and then reverse the output again
        #   such that the output from the layer is always from x1x1 to xnxn.
        bwd = cell(rnn, rnn_dim, only_return_final=False, backwards=True,
                   **cell_args)
        if add_dense:
            # reshape for dense
            fwd = ReshapeLayer(fwd, (-1, rnn_dim))
            bwd = ReshapeLayer(bwd, (-1, rnn_dim))
            fwd = DenseLayer(fwd, num_units=rnn_dim, nonlinearity=tanh)
            bwd = DenseLayer(bwd, num_units=rnn_dim, nonlinearity=tanh)
            # dropout
            fwd = dropout(fwd, p=dropout_p)
            bwd = dropout(bwd, p=dropout_p)
            # reshape back to input format
            fwd = ReshapeLayer(fwd, rnn_shape)
            bwd = ReshapeLayer(bwd, rnn_shape)
        # merge over lstm output dim (axis=2)
        rnn = ElemwiseSumLayer(incomings=[fwd, bwd])
    return rnn


class BiRNN(object):
    def __init__(self, emb_dim, rnn_dim, hid_dim, vocab_size, context,
                 cell='lstm', add_dense=True, dropout_p=0.2, depth=1,
                 prod=False, **cell_args):
        self.emb_dim = emb_dim
        self.rnn_dim = rnn_dim
        self.hid_dim = hid_dim
        self.vocab_size = vocab_size
        self.context = context
        self.cell = cell
        self.add_dense = add_dense
        self.depth = depth
        self.cell_args = cell_args

        self.prod = prod

        # Input is integer matrices (batch_size, seq_length)
        input_layer = InputLayer(shape=(None, context * 2),
                                 input_var=T.imatrix())
        self.emb_W = np.random.uniform(size=(vocab_size, emb_dim),
                                       low=-0.05,
                                       high=0.05).astype(np.float32)
        emb = EmbeddingLayer(input_layer, input_size=vocab_size,
                             output_size=emb_dim, W=self.emb_W)
        batch_size, _ = input_layer.input_var.shape
        rnn_shape = (batch_size, context * 2, rnn_dim)
        rnn = bid_layer(
            emb, rnn_dim, batch_size, rnn_shape, cell=cell,
            add_dense=add_dense, dropout_p=dropout_p, depth=depth, **cell_args)
        # time distributed dense
        output_shape = (batch_size, context * 2, hid_dim)
        rnn = ReshapeLayer(rnn, (-1, rnn_dim))
        rnn = DenseLayer(rnn, num_units=hid_dim)
        rnn = ReshapeLayer(dropout(rnn, p=dropout_p), output_shape)
        # flatten
        rnn = flatten(rnn)
        self.output = DenseLayer(
            rnn, num_units=vocab_size, nonlinearity=softmax)

        # Don't compile train and test functions in production mode
        if not prod:
            # T.nnet.categorical_crossentropy allows to represent true dist
            # as an integer vector (implicitely casting to a one-hot matrix)
            lr, targets = T.fscalar('lr'), T.ivector('targets')
            pred = get_output(self.output)
            loss = T.nnet.categorical_crossentropy(pred, targets).mean()
            params = get_all_params(self.output, trainable=True)
            updates = lasagne.updates.rmsprop(loss, params, lr)
            print("Compiling training function")
            self._train = theano.function(
                [input_layer.input_var, targets, lr],
                loss, updates=updates, allow_input_downcast=True)

            test_pred = get_output(self.output, deterministic=True)
            test_loss = T.nnet.categorical_crossentropy(test_pred, targets).mean()
            test_acc = accuracy(test_pred, targets)

            print("Compiling test function")
            self._test = theano.function(
                [input_layer.input_var, targets],
                [test_loss, test_acc], allow_input_downcast=True)

        print("Compiling predict function")
        if prod:
            pred = get_output(self.output, deterministic=True)
        else:
            pred = test_pred
        self._predict = theano.function(
            [input_layer.input_var], pred, allow_input_downcast=True)

    def train_on_batch(self, batch_X, batch_y, lr=0.01, shuffle=True):
        """
        Parameters:
        -----------
        batch_X: np.array(size=(batch_size, seq_len), dtype=np.int)
        batch_y: np.array(size=(batch_size, vocab_size))
        """
        if self.prod:
            raise ValueError("No training is possible in production mode")
        assert batch_X.shape[0] == batch_y.shape[0]
        if shuffle:
            p = np.random.permutation(batch_X.shape[0])
            return self._train(batch_X[p, :], batch_y[p], lr)
        else:
            return self._train(batch_X, batch_y, lr)

    def test_on_batch(self, batch_X, batch_y, **kwargs):
        """
        Parameters:
        -----------
        batch_X: np.array(size=(batch_size, seq_len), dtype=np.int)
        batch_y: np.array(size=(batch_size, vocab_size))
        """
        if self.prod:
            raise ValueError("No testing is possible in production mode")
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
        if self.prod:
            raise ValueError("No training is possible in production mode")
        for e in range(epochs):
            losses = []
            for b, (X, y) in enumerate(batch_gen(batch_size)):
                loss = self.train_on_batch(X, y, **kwargs)
                losses.append(loss)
                if b % batches == 0:
                    yield False, e, b, losses
            yield True, e, b, losses

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
        # save weights
        np.savez(prefix + "_weights.npz", *get_all_param_values(self.output))
        # save network config params
        with open(prefix + "_config.json", "w") as f:
            f.write(json.dumps(
                {"emb_dim": self.emb_dim,
                 "rnn_dim": self.rnn_dim,
                 "hid_dim": self.hid_dim,
                 "vocab_size": self.vocab_size,
                 "context": self.context,
                 "cell": self.cell,
                 "add_dense": self.add_dense,
                 "depth": self.depth,
                 "cell_args": self.cell_args}))

    @classmethod
    def load(cls, prefix, prod=True):
        import json
        with open(prefix + "_config.json") as data:
            params = json.load(data)
        network = cls(
            # positional params
            params.get('emb_dim', defs['emb_dim']),
            params.get('rnn_dim', defs['rnn_dim']),
            params.get('hid_dim', defs['hid_dim']),
            params['vocab_size'],
            params.get('context', defs['context']),
            # keyword params
            prod=prod,
            cell=params.get('cell', defs['cell']),
            add_dense=params.get('add_dense', defs['add_dense']),
            depths=params.get('depth', defs['depth']),
            # kwarg params
            **params.get('cell_args', {}))
        with np.load(prefix + '_weights.npz') as data:
            params = [data['arr_%d' % i] for i in range(len(data.files))]
            set_all_param_values(network.output, params)
        return network


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-R', '--depth', type=int, default=defs['depth'])
    parser.add_argument('-C', '--cell', type=str, default=defs['cell'])
    parser.add_argument('-m', '--emb_dim', type=int, default=defs['emb_dim'])
    parser.add_argument('-l', '--rnn_dim', type=int, default=defs['rnn_dim'])
    parser.add_argument('-H', '--hid_dim', type=int, default=defs['hid_dim'])
    parser.add_argument('-D', '--dropout', type=float, default=defs['dropout'])
    parser.add_argument('-f', '--add-dense', action='store_true',
                        default=defs['add_dense'])
    parser.add_argument('-c', '--context', type=float, default=defs['context'])
    parser.add_argument('-b', '--batch_size', type=int, default=1024)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-n', '--num_examples', type=int, default=10000000)
    parser.add_argument('-r', '--root', type=str, required=True)
    parser.add_argument('-p', '--model_prefix', type=str, required=True)
    parser.add_argument('-d', '--db', type=str, default='db.json')
    parser.add_argument('-L', '--loss', type=int, default=50,
                        help='report loss every l batches')
    parser.add_argument('-g', '--grad_clipping', type=int, default=0)
    parser.add_argument('-P', '--peepholes', action='store_true')

    from casket.nlp_utils import Corpus, Indexer
    from casket import Experiment as E
    import os
    import utils

    args = parser.parse_args()
    root = args.root
    path = args.db
    assert os.path.isdir(root), "Root path doesn't exist"

    CELL = args.cell
    BATCH_SIZE = args.batch_size
    NUM_BATCHES = int(args.num_examples / BATCH_SIZE)
    EPOCHS = args.epochs
    DEPTH = args.depth
    EMB_DIM = args.emb_dim
    RNN_DIM = args.rnn_dim
    HID_DIM = args.hid_dim
    DROPOUT = args.dropout
    CONTEXT = args.context
    ADD_DENSE = args.add_dense
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

    dev_test_batch_size = int(args.num_examples * 0.005)

    test = list(batch_gen(dev_test_batch_size, corpus=test))
    test_batch_choice = np.random.randint(len(test))
    test_X, test_y = test[test_batch_choice]
    print("test batch [%d]" % test_batch_choice)
    del test

    dev = list(batch_gen(dev_test_batch_size, corpus=dev))    
    dev_batch_choice = np.random.randint(len(dev))
    dev_X, dev_y = dev[dev_batch_choice]
    print("dev batch [%d]" % dev_batch_choice)
    del dev

    vocab_size = idxr.vocab_len()
    birnn = BiRNN(EMB_DIM, RNN_DIM, HID_DIM, vocab_size, CONTEXT,
                  depth=DEPTH, add_dense=ADD_DENSE, dropout_p=DROPOUT,
                  grad_clipping=args.grad_clipping, peepholes=args.peepholes)

    print("Starting training")
    db = E.use(path, exp_id='lasagne-birnn').model("")
    with db.session(vars(args), ensure_unique=False) as session:
        try:
            for flag, e, b, losses in birnn.fit(
                    batch_gen, EPOCHS, BATCH_SIZE, batches=LOSS):
                loss, acc = birnn.test_on_batch(dev_X, dev_y)
                loss, ascc = float(loss), float(acc)
                utils.log_batch(e, b, np.mean(losses), losses[-1], loss, acc)
                if flag:
                    print()
        except KeyboardInterrupt:
            print("Interrupted\n")
        finally:
            loss, acc = birnn.test_on_batch(test_X, test_y)
            print("Test loss [%f], test acc [%f]" % (float(loss), float(acc)))
            session.add_result({'test_acc': float(acc), 'test_loss': float(loss)})
            birnn.save(args.model_prefix)
            idxr.save(args.model_prefix + '_indexer.json')
