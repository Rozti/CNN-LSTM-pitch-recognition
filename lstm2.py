#!/usr/bin/env python
import glob
import os

import numpy

from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Dropout

from utils import  get_number, create_cnn_one_file

def get_notes(mid_dir, png_dir):
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes_tuple = []
    # for file in glob.glob('Magisterka/www.audiolabs-erlangen.de/content/resources/MIR/SMD/02-midi/data/*.mid'): #("data/*.mid"):
    for file in glob.glob(os.path.join(mid_dir, "*.mid")):
        # print("Parsing %s" % file)
        filename, number = get_number(file)

        png_name = os.path.split(file)[1].replace(".mid", ".png")
        img, one_hot = create_cnn_one_file(png_name, mid_dir=mid_dir, png_dir=png_dir)
        notes_tuple.append((filename, int(number), one_hot, img))
        # append NAME, SLICE_INDEX, ONE_HOT, IMG

    notes_tuple = sorted(notes_tuple, key=lambda tup: (tup[0], tup[1]))
    notes_tuple = list(filter(lambda x: max(x[2]) == 1, notes_tuple))
    return notes_tuple


def prepare_sequences(notes_tuple):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 36
    # get all pitch names
    # create a dictionary to map pitches to integers
    network_input = []
    network_output = []
    network_input_cnn = []

    song_set = set(item[0] for item in notes_tuple)
    for item in song_set:
        song_list = list(filter(lambda x: x[0] == item, notes_tuple))
        for i in range(0, len(song_list) - sequence_length, 1):
            sequence_in = []
            for j in range(i, i + sequence_length):
                #appending one hot -> notes_tuple[2]
                sequence_in.append(song_list[j][2])
            sequence_out = song_list[i + sequence_length][2]
            sequence_out_img = song_list[i + sequence_length][3]

            network_input.append(sequence_in)
            network_output.append(sequence_out)
            network_input_cnn.append(sequence_out_img)

    network_input = numpy.array(network_input)
    network_output = numpy.array(network_output)
    network_input_cnn = numpy.array(network_input_cnn)
    print(f"INPUT LSTM {network_input.shape}")
    print(f"INPUT CNN {network_input_cnn.shape}")
    print(f"OUTPUT {network_output.shape}")
    # print(network_input_cnn)
    return (network_input, network_output, network_input_cnn)


def prepare_data_cnn(notes_tuple):
    network_output = list(map(lambda x: x[2], notes_tuple))
    network_input_cnn = list(map(lambda x: x[3], notes_tuple))

    network_output = numpy.array(network_output)
    network_input_cnn = numpy.array(network_input_cnn)
    return network_output, network_input_cnn


def create_model(network_input):

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
    dense2 = Dense(128)(dropout2)

    """create CNN model"""
    img_x, img_y = 145, 49
    input_shape = (img_x, img_y, 4)

    cnn_Input = keras.Input(shape=input_shape, name="cnn")
    cnn_conv = keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                                   activation='tanh')(cnn_Input)

    cnn_drop1 = Dropout(0.5)(cnn_conv)
    cnn_max_pooling = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(cnn_drop1)
    cnn_conv2 = Conv2D(64, (3, 3), activation='tanh')(cnn_max_pooling)
    cnn_drop2 = Dropout(0.5)(cnn_conv2)
    cnn_max_pooling2 = MaxPooling2D(pool_size=(2, 2))(cnn_drop2)
    cnn_flat = Flatten()(cnn_max_pooling2)
    cnn_dense = Dense(128)(cnn_flat)

    """merge models"""
    concatenate = keras.layers.Concatenate()([dense2, cnn_dense])
    # concatenate = keras.layers.Average()([dense2, cnn_dense])
    merged_dense = Dense(128)(concatenate)
    merged_activation = Activation(activation='softmax', name="merged")(merged_dense)

    model = keras.Model(
        inputs=[lstm_input, cnn_Input],
        outputs=[merged_activation],
    )
    # keras.utils.plot_model(model, "merged_model2.png", show_shapes=True)
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=.0001, decay=1e-6),
                  metrics=['accuracy'])

    return model

def create_model_cnn():
    img_x, img_y = 145, 49
    input_shape = (img_x, img_y, 4)

    cnn_Input = keras.Input(shape=input_shape, name="cnn")
    cnn_conv = keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                                   activation='tanh')(cnn_Input)

    cnn_drop1 = Dropout(0.5)(cnn_conv)
    cnn_max_pooling = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(cnn_drop1)
    cnn_conv2 = Conv2D(64, (3, 3), activation='tanh')(cnn_max_pooling)
    cnn_drop2 = Dropout(0.5)(cnn_conv2)
    cnn_max_pooling2 = MaxPooling2D(pool_size=(2, 2))(cnn_drop2)
    cnn_flat = Flatten()(cnn_max_pooling2)
    cnn_dense = Dense(128)(cnn_flat)
    activation = Activation(activation='softmax', name="cnn_out")(cnn_dense)

    model = keras.Model(
        inputs=[cnn_Input],
        outputs=[activation],
    )
    # keras.utils.plot_model(model, "merged_model2.png", show_shapes=True)
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=.0001, decay=1e-6),
                  metrics=['accuracy'])

    return model


def start(target):
    mid_dir = target+"_mid"
    png_dir = target+"_wav"

    a = get_notes(mid_dir=mid_dir, png_dir=png_dir)
    network_input_lstm, network_output, network_input_cnn = prepare_sequences(a)

    filepath = "cnn+lstm2-weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]
    model = create_model(network_input_lstm)

    model.fit(
            {"lstm": network_input_lstm, "cnn": network_input_cnn},
            {"merged": network_output },
            epochs = 50, batch_size = 128, callbacks = callbacks_list)


def start_cnn(target):
    mid_dir = target+"_mid"
    png_dir = target+"_wav"

    a = get_notes(mid_dir=mid_dir, png_dir=png_dir)
    network_output, network_input_cnn = prepare_data_cnn(a)

    filepath = "cnn-weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]
    model = create_model_cnn()

    model.fit(
        {"cnn": network_input_cnn},
        {"cnn_out": network_output},
        epochs=400, batch_size=128, callbacks=callbacks_list)


#start()
#start_cnn()


