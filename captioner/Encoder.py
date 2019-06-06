
from keras import models, Sequential
from keras.layers import Dense, LSTM, Input, Bidirectional
from pathlib import Path


class Encoder:

    def __init__(self, input_shape, input_data_type, output_shape, output_data_type, n_unit, embedding=False):
        self.input = Input(shape=input_shape, dtype=input_data_type)
        self.output = LSTM(256, dtype=output_data_type, return_sequences=True)
        self.n_unit = n_unit
        self.model = self.encoder_body(embedding)
        self.s0 = 0
        self.c0 = 0

    def encoder_body(self, embedding):
        model = Sequential()
        model.add(self.input)
        for i in range(self.n_unit):
            model.add(Bidirectional(LSTM(256, return_sequences=True)))
        model.add(self.output)
        model.compile("Adam", loss="mean_squared_error")
        return model
        # generate lstm captioning

    def attention_layer(self):
        layer = Dense(32)
