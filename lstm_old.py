#!/usr/bin/env python
import glob
import os
import pickle

import numpy
import pydub
from PIL import Image
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from music21 import chord, note, converter, instrument
from pretty_midi import pretty_midi

import models as models
from utils import pretty_midi_to_one_hot, slice_to_categories

mid_dir = 'train/train_quad_mid'
mid_test_dir = 'test/test/test_quad_mid'
png_dir = 'train/train_quad_wav'


def set_file_to_1channel_wav(filename):
    if filename is None:
        return None
    my_sound = None
    if filename.endswith('.mp3'):
        my_sound = pydub.AudioSegment.from_mp3(filename)
    elif filename.endswith('wav'):
        my_sound = pydub.AudioSegment.from_wav(filename)
    my_sound = my_sound.set_channels(1)
    return my_sound


def get_number(string):
    tmp = string.replace(".mid", "")
    tmp = tmp.split("_")
    number = tmp.pop()
    tmp = "_".join(tmp)
    tmp += ".mid"
    return tmp, number


def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []
    notes_tuple = []
    filled = 0
    empty = 0
    
    dir_list = [glob.glob(os.path.join(mid_dir, "*.mid")), glob.glob(os.path.join(mid_test_dir, "*.mid"))]
    for dir in dir_list:
        for file in dir:
            # print("Parsing %s" % file)
            filename, number = get_number(file)
            try:
                midi = converter.parse(file)
                try:  # file has instrument parts
                    s2 = instrument.partitionByInstrument(midi)
                    notes_to_parse = s2.parts[0].recurse()
                    # print(f"NOTES TO PARSE {notes_to_parse}")
                except:  # file has notes in a flat structure
                    notes_to_parse = midi.flat.notes
                    # notes_to_parse.show('text')

                notes_tmp = []
                for element in notes_to_parse:

                    if isinstance(element, note.Note):
                        notes_tmp.append(element.pitch)
                    elif isinstance(element, chord.Chord):
                        for n in element.pitches:
                            notes_tmp.append(n)
                    else:
                        tmp = note.Rest  # ADDED
                        notes_tmp.append(tmp)

                notes.append('.'.join(str(n) for n in notes_tmp))
                notes_tuple.append((filename, int(number), '.'.join(str(n) for n in notes_tmp)))
                filled += 1
            except:
                empty += 1
                notes.append(' ')
                notes_tuple.append((filename, int(number), ' '))

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)
    # print(f"WRITTEN: {filled}, EMPTY {empty}")
    return notes, notes_tuple


def prepare_sequences(n_vocab, pitchnames, notes_tuple):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 36
    # get all pitch names
    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    network_input = []
    network_output = []
    network_input_cnn = []
    notes_tuple = list(filter(lambda x: x[2] != " ", notes_tuple))
    song_set = set(item[0] for item in notes_tuple)
    for item in song_set:
        song_list = list(filter(lambda x: x[0] == item, notes_tuple))
        for i in range(0, len(song_list) - sequence_length, 1):
            sequence_in = []
            for j in range(i, i + sequence_length):
                sequence_in.append(song_list[j][2])
            sequence_out = song_list[i + sequence_length][2]
            song_name = (song_list[i + sequence_length][0]).replace(".mid", "") + "_" + str(
                song_list[i + sequence_length][1]) + ".png"

            network_input_cnn.append(os.path.split(song_name)[1])
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])
    # create input sequences and the corresponding outputs

    n_patterns = len(network_input)
    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)
    network_output = np_utils.to_categorical(network_output)
    print(f"INPUT {network_input.shape}")
    print(f"OUTPUT {network_output.shape}")
    # print(network_input_cnn)
    return (network_input, network_output, network_input_cnn)


def create_cnn_input(network_input_cnn):
    x_train, y_train = [], []
    for name in network_input_cnn:
        x, y = create_cnn_one_file(name)
        x_train.append(x)
        y_train.append(y)
    x_train = numpy.array(x_train)
    x_train /= 255.0
    y_train = numpy.array(y_train)
    return x_train, y_train

def create_cnn_one_file(name):
    img_file = os.path.join(png_dir, name)
    mid_file = os.path.join(mid_dir, name.replace(".png", ".mid"))
    pm = pretty_midi.PrettyMIDI(mid_file)
    oh = pretty_midi_to_one_hot(pm)
    oh = slice_to_categories(oh)

    im = Image.open(img_file)
    im = im.crop((14, 13, 594, 301))
    resize = im.resize((49, 145), Image.NEAREST)
    resize.load()
    arr = numpy.asarray(resize, dtype="float32")

    # print(f'PNG {img_file} MIDI {mid_file}')
    return arr,oh

def main():

    notes, notes_tuple = get_notes()
    notes_tuple = sorted(notes_tuple, key=lambda tup: (tup[0], tup[1]))
    # print(notes_tuple)
    n_vocab = len(set(notes))
    pitchnames = sorted(set(item for item in notes))

    network_input_lstm, network_output_lstm, network_input_cnn_files = prepare_sequences(n_vocab, pitchnames, notes_tuple)
    network_input_cnn, network_output_cnn = create_cnn_input(network_input_cnn_files)

    print("SHAPES")
    print(f"LSTM IN: {network_input_lstm.shape}, OUT: {network_output_lstm.shape}")
    print(f"CNN IN: {network_input_cnn.shape}, OUT: {network_output_cnn.shape}")



    model = models.create_model(network_input_lstm,n_vocab)

    filepath = "cnn+lstm1-weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(
        {"lstm": network_input_lstm, "cnn": network_input_cnn},
        {"merged": network_output_cnn, "lstm_out": network_output_lstm, "cnn_out": network_output_cnn},
        epochs = 50, batch_size = 128, callbacks = callbacks_list)

main()
