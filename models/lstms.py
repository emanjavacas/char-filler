# coding: utf-8

from keras.models import Model
from keras.layers import Input, merge, Dense, Flatten
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import Bidirectional, TimeDistributed


def simple_lstm(n_chars, context=10, hidden_layer=128, **kwargs):
    in_layer = Input(shape=(context * 2, 1))
    lstm = LSTM(output_dim=hidden_layer, name='lstm')(in_layer)
    out_layer = Dense(output_dim=n_chars, activation='softmax')(lstm)
    model = Model(input=in_layer, output=out_layer)
    model.compile('rmsprop', loss='categorical_crossentropy', **kwargs)
    return model


def bilstm(n_chars, context=10, hidden_layer=128, rnn_layers=1, **kwargs):
    in_layer = Input(shape=(context * 2, 1))
    for i in range(rnn_layers):
        if i == 0:
            lstm = in_layer
        else:
            wrapped = LSTM(
                output_dim=hidden_layer, activation='tanh',
                return_sequences=True, name='bistm_%d' % i)
            lstm = Bidirectional(wrapped, merge_mode='sum')(lstm)
    flattened = Flatten()(lstm)
    hidden = Dense(n_chars, activation='tanh')(flattened)
    out_layer = Dense(n_chars, activation='softmax')(hidden)
    model = Model(input=in_layer, output=out_layer)
    model.compile('rmsprop', loss='categorical_crossentropy', **kwargs)
    return model


def emb_bilstm(n_chars, emb_dim,
               context=10, hidden_layer=128, rnn_layers=1, **kwargs):
    in_layer = Input(shape=(context * 2,), dtype='int32')
    emb_layer = Embedding(input_dim=context * 2, output_dim=emb_dim)(in_layer)
    for i in range(rnn_layers):
        if i == 0:
            lstm = emb_layer
        else:
            wrapped = LSTM(
                output_dim=hidden_layer, activation='tanh',
                return_sequences=True, name='bilstm_%d' % i)
            lstm = Bidirectional(wrapped, merge_mode='sum')
    # flattened = Flatten()(lstm)
    # dense = Dense(n_chars, activation='tanh')(flattened)
    dense = TimeDistributed(Dense(n_chars), name='dense')(lstm)
    out_layer = Dense(n_chars, activation='softmax', name='output')(dense)
    model = Model(input=emb_layer, output=out_layer)
    model.compile('rmsprop', loss='categorical_crossentropy', **kwargs)
    return model
