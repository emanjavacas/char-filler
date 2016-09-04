# coding: utf-8

from keras.models import Model
from keras.layers import Input, merge, Dense, Flatten
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding


def simple_lstm(n_chars, context=10, hidden_layer=128, **kwargs):
    in_layer = Input(shape=(context * 2, 1))
    lstm = LSTM(output_dim=hidden_layer, name='lstm')(in_layer)
    out_layer = Dense(output_dim=n_chars, activation='softmax')(lstm)
    model = Model(input=in_layer, output=out_layer)
    model.compile('rmsprop', loss='categorical_crossentropy', **kwargs)
    return model


def bilstm(n_chars, context=10, hidden_layer=128, rnn_layers=1, **kwargs):
    in_layer = Input(shape=(context * 2, 1))
    lstm = None
    for i in range(rnn_layers):
        if i == 0:
            curr_input = in_layer
        else:
            curr_input = lstm
        l2r = LSTM(hidden_layer, return_sequences=True,
                   dropout_W=0.2, dropout_U=0.2)(curr_input)
        r2l = LSTM(hidden_layer, return_sequences=True, go_backwards=True,
                   dropout_W=0.2, dropout_U=0.2)(curr_input)
        lstm = merge([l2r, r2l], mode='sum', name='bilstm_%d' % i)
    hidden = Dense(hidden_layer, activation='tanh')(Flatten()(lstm))
    out_layer = Dense(n_chars, activation='softmax')(hidden)
    model = Model(input=in_layer, output=out_layer)
    model.compile('rmsprop', loss='categorical_crossentropy', **kwargs)
    return model


def emb_bilstm(n_chars, emb_dim,
               context=10, hidden_layer=128, rnn_layers=1, **kwargs):
    in_shape = context * 2
    in_layer = Input(shape=(in_shape,), dtype='int32', name='input')
    emb_layer = Embedding(
        input_dim=n_chars, output_dim=emb_dim, input_dtype='int32',
        name='emb')(in_layer)
    lstm = None
    for i in range(rnn_layers):
        if i == 0:
            curr_input = emb_layer
        else:
            curr_input = lstm
        l2r = LSTM(hidden_layer, return_sequences=True,
                   dropout_W=0.2, dropout_U=0.2)(curr_input)
        r2l = LSTM(hidden_layer, return_sequences=True, go_backwards=True,
                   dropout_W=0.2, dropout_U=0.2)(curr_input)
        lstm = merge([l2r, r2l], mode='sum', name='bilstm_%d' % i)
    hidden = Dense(hidden_layer, activation='tanh')(Flatten()(lstm))
    out_layer = Dense(n_chars, activation='softmax')(hidden)
    model = Model(input=in_layer, output=out_layer)
    model.compile('rmsprop', loss='categorical_crossentropy', **kwargs)
    return model
