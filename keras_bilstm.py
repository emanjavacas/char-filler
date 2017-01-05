# coding: utf-8

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import Bidirectional, TimeDistributed

import numpy as np

def simple_lstm(n_chars, context=10, hidden_layer=128):
    in_layer = Input(shape=(context * 2, 1))
    lstm = LSTM(output_dim=hidden_layer, name='lstm')(in_layer)
    out_layer = Dense(output_dim=n_chars, activation='softmax')(lstm)
    model = Model(input=in_layer, output=out_layer)
    return model


def bilstm_layer(input_layer, lstm_dims, rnn_layers, dropout):
    lstm = None
    if isinstance(lstm_dims, (list, tuple)):
        lstm_dims = lstm_dims
    else:
        assert isinstance(lstm_dims, int)
        lstm_dims = [lstm_dims] * rnn_layers
    for i in range(rnn_layers):
        if i == 0:
            nested = input_layer
        else:
            nested = lstm
        wrapped = LSTM(
            output_dim=lstm_dims[i], activation='tanh', return_sequences=True,
            dropout_W=dropout, dropout_U=dropout, name='bistm_%d' % i)
        lstm = Bidirectional(wrapped, merge_mode='sum')(nested)
    return lstm


def bilstm(n_chars, context=10, lstm_dims=128, hidden_dim=250, rnn_layers=1,
           dropout=0.0):
    in_layer = Input(shape=(context * 2, n_chars), name='input')
    lstm = bilstm_layer(in_layer, lstm_dims, rnn_layers, dropout=dropout)
    dense = TimeDistributed(Dense(hidden_dim), name='dense')(lstm)
    flattened = Flatten(name='flattened')(dense)
    out_layer = Dense(n_chars, activation='softmax')(flattened)
    model = Model(input=in_layer, output=out_layer)
    return model


def emb_bilstm(n_chars, emb_dim, context=10, lstm_dims=128, hidden_dim=250,
               rnn_layers=1, dropout=0.0):
    in_layer = Input(shape=(context * 2,), dtype='int32', name='input')
    emb_layer = Embedding(
        input_dim=n_chars, output_dim=emb_dim, input_dtype='int32',
        name='emb')(in_layer)
    lstm = bilstm_layer(emb_layer, lstm_dims, rnn_layers, dropout=dropout)
    dense = TimeDistributed(Dense(hidden_dim), name='dense')(lstm)
    flattened = Flatten(name='flattened')(dense)
    out_layer = Dense(n_chars, activation='softmax', name='output')(flattened)
    model = Model(input=in_layer, output=out_layer)
    return model


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('-o', '--optimizer', type=str, default='rmsprop')
    parser.add_argument('-R', '--rnn_layers', type=int, default=1)
    parser.add_argument('-m', '--emb_dim', type=int, default=28)
    parser.add_argument('-l', '--lstm_dim', type=int, default=100)
    parser.add_argument('-H', '--hidden_dim', type=int, default=250)
    parser.add_argument('-D', '--dropout', type=float, default=0.0)
    parser.add_argument('-b', '--batch_size', type=int, default=50)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-n', '--num_examples', type=int, default=10000000)    
    parser.add_argument('-r', '--root', type=str, required=True)
    parser.add_argument('-p', '--model_prefix', type=str, required=True)
    parser.add_argument('-d', '--db', type=str, default='db.json')
    parser.add_argument('-L', '--loss', type=int, default=1,
                        help='report loss every l batches')

    from keras.utils.np_utils import to_categorical
    from casket.nlp_utils import Corpus, Indexer
    from casket import Experiment as E
    import utils
    import os, itertools
    
    args = parser.parse_args()
    root = args.root
    path = args.db
    assert os.path.isdir(root), "Root path doesn't exist"

    BATCH_SIZE = args.batch_size
    NUM_BATCHES = int(args.num_examples / BATCH_SIZE)
    EPOCHS = args.epochs
    OPTIMIZER = args.optimizer
    RNN_LAYERS = args.rnn_layers
    EMB_DIM = args.emb_dim
    LSTM_DIM = args.lstm_dim
    HIDDEN_DIM = args.hidden_dim
    DROPOUT = args.dropout

    idxr = Indexer(pad='~', oov='±')
    train = Corpus(os.path.join(root, 'train'))
    test = Corpus(os.path.join(root, 'test'))
    dev = Corpus(os.path.join(root, 'dev'))

    print("Building encoder on train corpus")
    idxr.fit(train.chars())  # quick pass to fit vocab
    n_chars = idxr.vocab_len()

    def build_set(corpus, vocab_len, size=2000, one_hot_enc=True):
        dataset = itertools.islice(
            corpus.generate(indexer=idxr, fitted=True), size)
        X, y = list(zip(*dataset))
        X = np.asarray(X)
        if one_hot_enc:
            X = one_hot(X, vocab_len)
        y = to_categorical(y, nb_classes=vocab_len)
        return X, y

    print("Encoding test set")
    has_emb = args.model == "emb_bilstm"
    X_test, y_test = build_set(test, n_chars, one_hot_enc=not has_emb)

    print("Encoding dev set")
    X_dev, y_dev = build_set(dev, n_chars, one_hot_enc=not has_emb)

    print("Compiling model")
    if args.model == 'bilstm':
        model = bilstm(
            n_chars,
            rnn_layers=RNN_LAYERS, lstm_dims=LSTM_DIM,
            hidden_dim=HIDDEN_DIM, dropout=DROPOUT)

    elif args.model == 'emb_bilstm':
        model = emb_bilstm(
            n_chars, EMB_DIM,
            rnn_layers=RNN_LAYERS, lstm_dims=LSTM_DIM,
            hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
    else:
        raise ValueError("Missing model [%s]" % args.model)

    model.compile(OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    print("Starting training")
    db = E.use(path, exp_id="char-fill").model(args.model)
    with db.session(vars(args), ensure_unique=False) as session:
        try:
            from time import time
            start = time()
            for e in range(EPOCHS):
                losses = []
                batches = train.generate_batches(indexer=idxr, batch_size=BATCH_SIZE)
                for b, (X, y) in enumerate(itertools.islice(batches, NUM_BATCHES)):
                    X = np.asarray(X) if has_emb else one_hot(X, n_chars)
                    y = to_categorical(y, nb_classes=n_chars)
                    loss, _ = model.train_on_batch(X, y)
                    losses.append(loss)
                    if b % args.loss == 0:
                        dev_loss, dev_acc = model.test_on_batch(X_dev, y_dev)
                        mean_loss, last_loss = np.mean(losses), losses[-1]
                        utils.log_batch(e, b, mean_loss, last_loss, dev_loss, dev_acc)
                print()
                session.add_epoch(
                    e, {'training_loss': str(np.mean(losses)),
                        'dev_loss': str(dev_loss),
                        'dev_acc': str(dev_acc)})
        except KeyboardInterrupt:
            print("Interrupted")
        finally:
            _, test_acc = model.test_on_batch(X_test, y_test)
            print("Test acc [%.4f]\n" % test_acc)
            session.add_result({'test_acc': str(test_acc)})
            session.add_meta({'run_time': time() - start})
            model.save_weights(args.model_prefix + '_weights.h5')
            utils.dump_json(model.get_config(), args.model_prefix + '_config.json')
            idxr.save(args.model_prefix + '_indexer.json')
