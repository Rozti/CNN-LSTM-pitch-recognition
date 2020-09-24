#!/usr/bin/env python
import lstm2 as lstm
import numpy as np

mid_dir = 'test/test/test_quad_mid'
png_dir = 'test/test/test_quad_wav'

def switch_demo(argument):
    switcher = {
        0: "C",
        1: "C#",
        2: "D",
        3: "D#",
        4: "E",
        5: "F",
        6: "F#",
        7: "G",
        8: "G#",
        9: "A",
        10: "A#",
        11: "B",
        12: "B#"
    }
    return switcher.get(argument, "Invalid note")


def midi_notes_to_letters(x):
    note, octave = x % 12, int(x / 12)
    return f"{switch_demo(note)}{octave}"

def load_model_merged(filepath, notes_tuple):
    network_input_lstm, network_output, network_input_cnn = lstm.prepare_sequences(notes_tuple)
    model = lstm.create_model(network_input_lstm)
    model.load_weights(filepath)
    return model

def merged_predict(mid_dir, png_dir, filepath):
    a = lstm.get_notes(mid_dir=mid_dir, png_dir=png_dir)
    network_input_lstm, network_output, network_input_cnn = lstm.prepare_sequences(a)

    # filepath = "cnn+lstm2-weights-improvement-49-0.0472-bigger.hdf5"
    model = lstm.create_model(network_input_lstm)
    model.load_weights(filepath)

    a = model.predict({"lstm": network_input_lstm, "cnn": network_input_cnn},batch_size = 128)

    result = 0
    result_one_note = 0
    one_note = 0
    two_notes = 0
    result_two_notes = 0
    many_notes = 0
    result_many_notes = 0
    for i in range(network_output.shape[0]):
        # print("##########")
        truth = []
        # print(network_output_cnn[i])
        for j in range(0, len(network_output[i])):
            if network_output[i][j] == 1:
                truth.append(j)
                # print(f"{midi_notes_to_letters(j)}")
        y = a[i]
        K  = len(truth)
        indexes = np.argpartition(y,-K)[-K:]
        # print(indexes)
        # print(list(map(lambda x: x in truth, indexes)))
        tmp =list(map(lambda x: x in truth, indexes))
        count = 0
        for element in tmp:
            if element is True:
                count += 1
        avg = count/K
        # print(avg)
        result += avg
        if K == 1:
            one_note +=1
            result_one_note+=avg
        elif K == 2:
            two_notes +=1
            result_two_notes +=avg
        else:
            many_notes +=1
            result_many_notes +=avg
        # for element in indexes:
        #     print(f"{midi_notes_to_letters(element)} : {y[element]}")
    print("CNN+LSTM2")
    print(f"RESULT {result / network_output.shape[0]}")
    print(f"RESULT ONE NOTE {result_one_note / one_note}")
    print(f"RESULT TWO NOTES {result_two_notes / two_notes}")
    print(f"RESULT MANY NOTES {result_many_notes / many_notes}")


def cnn_predict(mid_dir, png_dir, filepath):
    a = lstm.get_notes(mid_dir=mid_dir, png_dir=png_dir)
    network_output, network_input_cnn = lstm.prepare_data_cnn(a)

    # filepath = "cnn-weights-improvement-150-0.0415-bigger.hdf5"
    #"cnn-weights-improvement-50-0.0476-bigger.hdf5"
    model = lstm.create_model_cnn()
    model.load_weights(filepath)

    a = model.predict({"cnn": network_input_cnn}, batch_size=128)

    result = 0
    result_one_note = 0
    one_note = 0
    two_notes = 0
    result_two_notes = 0
    many_notes = 0
    result_many_notes = 0
    for i in range(network_output.shape[0]):
        # print("##########")
        truth = []
        # print(network_output_cnn[i])
        for j in range(0, len(network_output[i])):
            if network_output[i][j] == 1:
                truth.append(j)
                # print(f"{midi_notes_to_letters(j)}")
        y = a[i]
        K = len(truth)

        indexes = np.argpartition(y, -K)[-K:]
        # print(indexes)
        # print(list(map(lambda x: x in truth, indexes)))
        tmp = list(map(lambda x: x in truth, indexes))
        count = 0
        for element in tmp:
            if element is True:
                count += 1

        avg = count / K
        # print(avg)
        result += avg
        if K == 1:
            one_note += 1
            result_one_note += avg
        elif K == 2:
            two_notes += 1
            result_two_notes += avg
        else:
            many_notes += 1
            result_many_notes += avg

        # for element in indexes:
            # print(f"{midi_notes_to_letters(element)} : {y[element]}")
    print("CNN")
    print(f"RESULT {result / network_output.shape[0]}")
    print(f"RESULT ONE NOTE {result_one_note / one_note}")
    print(f"RESULT TWO NOTES {result_two_notes / two_notes}")
    print(f"RESULT MANY NOTES {result_many_notes / many_notes}")


merged_predict()
cnn_predict()
