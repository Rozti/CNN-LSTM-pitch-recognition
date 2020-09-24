

import numpy
from pretty_midi import pretty_midi
import numpy as np

from PIL import Image
import pretty_midi
import os, os.path

def pretty_midi_to_one_hot(pm, fs=100):
    """Compute a one hot matrix of a pretty midi object
    Parameters
    ----------
    pm : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    Returns
    -------
    one_hot : np.ndarray, shape=(128,times.shape[0])
        Piano roll of this instrument. 1 represents Note Ons,
        -1 represents Note offs, 0 represents constant/do-nothing
    """

    # Allocate a matrix of zeros - we will add in as we go
    one_hots = []

    if len(pm.instruments) < 1:
        return 0

    for instrument in pm.instruments:
        one_hot = np.zeros((128, int(fs*instrument.get_end_time())+1))
        for note in instrument.notes:
            # note on
            one_hot[note.pitch, int(note.start*fs)] = 1
            # print('note on',note.pitch, int(note.start*fs))
            # note off
            one_hot[note.pitch, int(note.end*fs)] = 0
            # print('note off',note.pitch, int(note.end*fs))
        one_hots.append(one_hot)

    one_hot = np.zeros((128, np.max([o.shape[1] for o in one_hots])))
    for o in one_hots:
        one_hot[:, :o.shape[1]] += o

    one_hot = np.clip(one_hot,-1,1)
    return one_hot


def slice_to_categories(piano_roll):
    notes_list = np.zeros(128)
    notes = np.nonzero(piano_roll)[0]
    notes = np.unique(notes)

    for note in notes:
        notes_list[note] = 1

    return notes_list

def create_cnn_one_file(name, png_dir, mid_dir):
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

def get_number(string):
    tmp = string.replace(".mid", "")
    tmp = tmp.split("_")
    number = tmp.pop()
    tmp = "_".join(tmp)
    tmp += ".mid"
    return tmp, number