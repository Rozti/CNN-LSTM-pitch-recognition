#!/usr/bin/env python
import glob
import os
from functools import reduce
from pathlib import Path
import pickle as pkl
import matplotlib.pyplot as plt
from utils import get_number, create_cnn_one_file


def get_notes(mid_dir, png_dir):
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes_tuple = []
    for file in glob.glob(os.path.join(mid_dir, f"*.mid")):
        # print("Parsing %s" % file)
        filename, number = get_number(file)

        png_name = os.path.split(file)[1].replace(".mid", ".png")
        img, one_hot = create_cnn_one_file(png_name, mid_dir=mid_dir, png_dir=png_dir)
        notes_tuple.append((filename, int(number), one_hot, img))
        # append NAME, SLICE_INDEX, ONE_HOT, IMG

    notes_tuple = sorted(notes_tuple, key=lambda tup: (tup[0], tup[1]))
    return notes_tuple

def save(filename, tab):

    with open(filename, 'wb') as f:
        pkl.dump(tab, f)


def load(filename):
    with open(filename, 'rb') as f:
        data = pkl.load(f)
    return data

def analyse_song(song_list, positive, negative):
    vector = [0]*128
    pitchnames_vector = [0] * 12
    chords = [0] * 7
    for item in song_list:
        how_many_notes = reduce(lambda a,b : a+b, item[2])
        how_many_notes = int(how_many_notes)
        positive += how_many_notes
        negative += (128-how_many_notes)
        if how_many_notes > 5:
            chords[6]+=1
        else:
            chords[how_many_notes] +=1
        vector = list(map(sum, zip(vector,item[2]))) # vector 128, count of pitches
    for i in range (0,12):
        pitchnames_vector[i] = reduce(lambda a,b : a+b, vector[i::12])

    pitches_count = reduce(lambda a,b : a+b, pitchnames_vector)

    vector_divided = list(map(lambda x: x/pitches_count, vector))
    pitchnames_vector_divided = list(map(lambda x: x/pitches_count, pitchnames_vector))
    chords_divided = list(map(lambda x: x/len(song_list), chords))
    directory = song_list[0][0].replace('.mid', '')
    tmp_name = os.path.split(directory)
    tmp_name = tmp_name[len(tmp_name) - 1]
    Path("plots").mkdir(parents=True, exist_ok=True)
    Path(f'plots/{tmp_name}').mkdir(parents=True, exist_ok=True)

    # count of all pitches
    plt.plot(vector_divided)
    tmp_name = os.path.split(directory)
    tmp_name = tmp_name[len(tmp_name)-1]
    plt.title(f'{tmp_name}')
    plt.suptitle(f'pitch distribution')
    plt.ylabel("%")
    plt.xlabel("pitch")
    plt.savefig(f'plots/{tmp_name}/all_pitches_percentage.png')
    plt.show()

    # count of pitchnamed pitches
    labels = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "H"]
    plt.bar(labels, pitchnames_vector_divided)

    plt.title(f'{tmp_name}')
    plt.suptitle(f'pitchname distribution')
    plt.ylabel("%")
    plt.xlabel("pitchname")
    plt.savefig(f'plots/{tmp_name}/pitchname_percentage.png')
    plt.show()

    #count of how many notes played at once
    labels = ["0", "1", "2", "3", "4", "5", ">5"]
    plt.bar(labels, chords_divided)
    plt.title(f'{tmp_name}')
    plt.suptitle(f'different pitch number distribution')
    plt.ylabel("%")
    plt.xlabel("number of notes at once")
    plt.savefig(f'plots/{tmp_name}/chords_percentage.png')
    plt.show()

    return positive,negative


def analyse_songs(pickle):
    data = load(pickle)
    song_set = set(item[0] for item in data)
    positive = 0
    negative = 0
    for item in song_set:
        song = list(filter(lambda x: x[0] == item, data))
        p_temp, n_temp =analyse_song(song, positive, negative)
        positive = p_temp
        negative = n_temp

    print(f"POSITIVE: {positive}, NEGATIVE: {negative}")
    print(f"POSITIVE: {positive/(positive+negative)}%, NEGATIVE: {negative/(positive+negative)}%")


# data = get_notes()
# save("data2.pickle", data)
# analyse_songs("data2.pickle")
