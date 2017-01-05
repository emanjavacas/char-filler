# coding: utf-8


from keras_bilstm import emb_bilstm
from corpus import Indexer, pad

import numpy as np


def load_emb_bilstm(n_chars, emb_dim, **kwargs):
    model = emb_bilstm(n_chars, emb_dim, **kwargs)
    model.load_weights('fitted/emb_bilstm_weights.h5')
    return model


def get_max_n(arr, max_n=1):
    return arr.argsort()[-max_n:][::-1]


class CharFiller(object):
    def __init__(self, model, idxr, context):
        self.model = model
        self.idxr = idxr
        self.context = context

    def predict(self, s, pos, with_prob=False, max_n=1):
        pad_code = self.idxr.pad_code
        left = self.idxr.transform(s[max(0, pos - self.context): pos])
        left = pad(left, self.context, paditem=self.idxr.pad_code)
        right = self.idxr.transform(s[pos+1: min(len(s), pos+self.context)])
        right = pad(right, self.context, paditem=pad_code, paddir='right')
        hist = np.asarray(left + right).reshape((1, self.context * 2))
        pred = self.model.predict(hist)[0]  # model returns a embedded array
        best = get_max_n(pred, max_n=max_n)
        if with_prob:
            return [(self.idxr.decode(code), pred[code]) for code in best]
        else:
            return [self.idxr.decode(code) for code in best]


if __name__ == '__main__':
    # load charfiller
    idxr = Indexer.load('fitted/emb_bilstm_indexer.json')
    m = load_emb_bilstm(idxr.vocab_len(), 28,
                        lstm_layer=250, hidden_layer=150, rnn_layers=3)
    filler = CharFiller(m, idxr, 10)
    filler.predict("this is a sentence to be filled with a lot of characters",
                   15, max_n=5, with_prob=True)
