#!/usr/bin/env python

import os
from pathlib import Path

import pydub
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def cqt(wav_file):
  # make CQT and save
  plt.figure(figsize=(7.5, 3.75))
  y, sr = librosa.load(wav_file)
  C = librosa.cqt(y, sr=sr)
  librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
                           sr=sr)
  plt.axis('off')
  plt.savefig(wav_file.replace(".wav", ".png"), bbox_inches="tight")
  plt.close('all')

def set_file_to_1channel_wav( filename):
  if filename is None:
    return None
  my_sound = None
  if filename.endswith('.mp3'):
    my_sound = pydub.AudioSegment.from_mp3(filename)
  elif filename.endswith('wav'):
    my_sound = pydub.AudioSegment.from_wav(filename)
  my_sound = my_sound.set_channels(1)
  return my_sound

def split_wav(mp3_file, target_dir, start_miliseconds, end_miliseconds, i, newAudio): #ms
  song_name = os.path.split(mp3_file)[-1][:-4]
  newAudio_length_miliseconds = newAudio.duration_seconds * 1000
  newAudio2 = newAudio[start_miliseconds : min(newAudio_length_miliseconds, end_miliseconds)]
  save_name = os.path.join(target_dir, song_name +  f"_{i}.wav")
  newAudio2.export(save_name, format="wav")
  cqt(save_name)

def split_midi(mid_file, target_dir, default_tempo=500000, target_segment_len=1.0):

  import mido
  from mido import MidiFile, MidiTrack, Message, MetaMessage
  song_name = os.path.split(mid_file)[-1][:-4]
  mid = MidiFile(mid_file)

  # identify the meta messages
  metas = []
  tempo = default_tempo
  for msg in mid:
    if msg.type is 'set_tempo':
      tempo = msg.tempo
    if msg.is_meta:
      metas.append(msg)
  for meta in metas:
    meta.time = int(mido.second2tick(meta.time, mid.ticks_per_beat, tempo))

  target = MidiFile()
  track = MidiTrack()
  track.extend(metas)
  target.tracks.append(track)
  prefix = 0
  time_elapsed = 0
  absolute_time = 0
  for msg in mid:
    # Skip non-note related messages
    if msg.is_meta:
      continue
    time_elapsed += msg.time
    if msg.type is not 'end_of_track':
      msg.time = int(mido.second2tick(msg.time, mid.ticks_per_beat, tempo))
      track.append(msg)
    #print(f'{msg} time1: {time_elapsed}')
    if msg.type is 'end_of_track' or time_elapsed >= target_segment_len:

      track.append(MetaMessage('end_of_track'))
      Path(os.path.join(target_dir + '_mid')).mkdir(parents=True, exist_ok=True)
      target.save(os.path.join(target_dir + '_mid', song_name + f'_{prefix}.mid'))
      # print(f"END OF A TRACK {absolute_time} {(absolute_time + time_elapsed)}")
      newAudio = set_file_to_1channel_wav(mid_file.replace(".mid", ".mp3"))

      Path(os.path.join(target_dir + '_wav')).mkdir(parents=True, exist_ok=True)
      split_wav(mid_file.replace(".mid", ".mp3"),
                target_dir + '_wav', absolute_time * 1000, (absolute_time + time_elapsed) * 1000, prefix, newAudio)
      absolute_time += time_elapsed

      target = MidiFile()
      track = MidiTrack()
      track.extend(metas)
      target.tracks.append(track)
      time_elapsed = 0
      prefix += 1

def main():
  target = "train/train_quad"
  directory = f'Magisterka/www.audiolabs-erlangen.de/content/resources/MIR/SMD/02-midi/data'
  for filename in os.listdir(directory):
      if filename.endswith(".mid"):
          midfile = os.path.join(directory, filename)
          print(midfile)
          split_midi(midfile, target, target_segment_len=0.250)

# main()
