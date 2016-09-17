# coding: utf-8

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import Bidirectional, TimeDistributed


def simple_lstm(n_chars, context=10, hidden_layer=128):
    in_layer = Input(shape=(context * 2, 1))
    lstm = LSTM(output_dim=hidden_layer, name='lstm')(in_layer)
    out_layer = Dense(output_dim=n_chars, activation='softmax')(lstm)
    model = Model(input=in_layer, output=out_layer)
    return model


def bilstm_layer(input_layer, lstm_layer, rnn_layers):
    lstm = None
    for i in range(rnn_layers):
        if i == 0:
            nested = input_layer
        else:
            nested = lstm
        wrapped = LSTM(
            output_dim=lstm_layer, activation='tanh',
            return_sequences=True, name='bistm_%d' % i)
        lstm = Bidirectional(wrapped, merge_mode='sum')(nested)
    return lstm


def bilstm(n_chars, context=10, lstm_layer=128, hidden_layer=250, rnn_layers=1):
    in_layer = Input(shape=(context * 2, n_chars), name='input')
    lstm = bilstm_layer(in_layer, lstm_layer, rnn_layers)
    dense = TimeDistributed(hidden_layer, name='dense')(lstm)
    flattened = Flatten(name='flattened')(dense)
    out_layer = Dense(n_chars, activation='softmax')(flattened)
    model = Model(input=in_layer, output=out_layer)
    return model


def emb_bilstm(n_chars, emb_dim, context=10, lstm_layer=128, hidden_layer=250, rnn_layers=1):
    in_layer = Input(shape=(context * 2,), dtype='int32', name='input')
    emb_layer = Embedding(
        input_dim=n_chars, output_dim=emb_dim, input_dtype='int32',
        name='emb')(in_layer)
    lstm = bilstm_layer(emb_layer, lstm_layer, rnn_layers)
    dense = TimeDistributed(Dense(hidden_layer), name='dense')(lstm)
    flattened = Flatten(name='flattened')(dense)
    out_layer = Dense(n_chars, activation='softmax', name='output')(flattened)
    model = Model(input=in_layer, output=out_layer)
    return model