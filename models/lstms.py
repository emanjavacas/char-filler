# coding: utf-8

from keras.models import Model
from keras.layers import Input, merge, Dense, TimeDistributed
from keras.layers.recurrent import LSTM


def simple_lstm(n_chars, context=10, hidden_layer=128):
    in_layer = Input(shape=(context * 2,))
    lstm = LSTM(output_dim=hidden_layer, name='lstm')(in_layer)
    out_layer = Dense(output_dim=n_chars, activation='softmax')(lstm)
    model = Model(input=in_layer, output=out_layer)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model


def bilstm(n_chars, context=10, hidden_layer=128, rnn_layers=1):
    in_layer = Input(shape=(context * 2,))
    bilstm = None
    for i in range(rnn_layers):
        if i == 0:
            curr_input = in_layer
        else:
            curr_input = bilstm
        l2r = LSTM(output_dim=hidden_layer, return_sequences=True)(curr_input)
        r2l = LSTM(output_dim=hidden_layer,
                   return_sequences=True,
                   go_backwards=True)(curr_input)
        bilstm = merge([l2r, r2l], mode='concat', name='bilstm_%d' % i)
    hidden = TimeDistributed(Dense(n_chars, activation='tanh'))(bilstm)
    out_layer = TimeDistributed(Dense(n_chars, activation='softmax'))(hidden)
    model = Model(input=in_layer, output=out_layer)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model
