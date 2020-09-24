#!/usr/bin/env python

from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Dropout


def create_model(network_input, n_vocab):
    """ create the structure of the neural network """

    """ create the LSMT model """
    lstm_input = keras.Input(shape=(network_input.shape[1], network_input.shape[2]), name="lstm")
    lstm = LSTM(512, recurrent_dropout=0.3, return_sequences=True)(lstm_input)
    lstm2 = LSTM(512, return_sequences=True, recurrent_dropout=0.3,)(lstm)
    lstm3 = LSTM(512)(lstm2)
    batch_norm = BatchNorm()(lstm3)
    dropout = Dropout(0.3)(batch_norm)
    dense = Dense(256)(dropout)
    activation = Activation('relu')(dense)
    batch_norm2 = BatchNorm()(activation)
    dropout2 = Dropout(0.3)(batch_norm2)
    dense2 = Dense(n_vocab)(dropout2)



    """create CNN model"""
    img_x, img_y = 145, 49
    input_shape = (img_x, img_y, 4)
    num_classes = 128

    cnn_Input = keras.Input(shape=input_shape, name="cnn")
    cnn_conv = keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                                   activation='tanh')(cnn_Input)

    cnn_drop1 = Dropout(0.5)(cnn_conv)
    cnn_max_pooling = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(cnn_drop1)
    cnn_conv2 = Conv2D(64, (3, 3), activation='tanh')(cnn_max_pooling)
    cnn_drop2 = Dropout(0.5)(cnn_conv2)
    cnn_max_pooling2 = MaxPooling2D(pool_size=(2, 2))(cnn_drop2)
    cnn_flat = Flatten()(cnn_max_pooling2)
    cnn_dense = Dense(num_classes)(cnn_flat)


    """merge models"""
    concatenate = keras.layers.Concatenate()([dense2, cnn_dense])
    merged_dense = Dense(num_classes)(concatenate)
    merged_activation = Activation(activation='relu', name="merged")(merged_dense)

    model = keras.Model(
        inputs=[lstm_input, cnn_Input],
        outputs=[merged_activation],
    )
    keras.utils.plot_model(model, "cnn+lstm1_model.png", show_shapes=True)
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=.0001, decay=1e-6),
                  metrics=['accuracy'])

    return model