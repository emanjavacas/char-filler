# coding: utf-8


from lstms import emb_bilstm
from corpus import Indexer, pad

import numpy as np


def load_emb_bilstm(n_chars, emb_dim, **kwargs):
    model = emb_bilstm(n_chars, emb_dim, **kwargs)
    model.load_weights('fitted/emb_bilstm_weights.h5')
    return model


class CharFiller(object):
    def __init__(self, model, indexer, context):
        self.model = model
        self.indexer = indexer
        self.context = context

    def predict(self, s, pos):
        left = self.indexer.transform(s[max(0, pos - self.context): pos])
        right = self.indexer.transform(s[pos + 1: min(len(s), pos + self.context)])
        left = pad(left, self.context, paditem=self.indexer.pad_code)
        right = pad(right, self.context, paditem=self.indexer.pad_code, paddir='right')
        hist = np.asarray(left + right).reshape((1, self.context * 2))
        return self.indexer.decode(np.argmax(self.model.predict(hist)))


if __name__ == '__main__':
    # load charfiller
    idxr = Indexer.load('fitted/emb_bilstm_indexer.json')
    m = load_emb_bilstm(idxr.vocab_len(), 28, lstm_layer=250, hidden_layer=150, rnn_layers=3)
    filler = CharFiller(m, idxr, 10)

    
