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
