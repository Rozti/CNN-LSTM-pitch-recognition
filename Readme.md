# CNN + LSTM based pitch recognition

Project shows the difference in music recognition by neural networsks. CNN based model and CNN + LSTM based model was used.

## Used packages
Project uses python 3.8.2 and packages listed below:

* Keras (2.3.1)
* numpy (1.18.4)
* matplotlib (3.1.1)
* librosa (0.7.2)
* pydub (0.23.1)
* mido (1.2.9)
* music21 (5.7.2))

## Usage
To use package You need to obtain dataset containing songs, both .mid files and .mp3 files. One used for this project is from [audiolabs erlangen](https://www.audiolabs-erlangen.de/resources/MIR/SMD/midi).

The main.py file is main file to be run. It has 4 different commands:

* main.py train [-h] --model {lstm,cnn} [--data DATA]
   * -h, --help:          show this help message and exit
   * --model {lstm,cnn}
   * --data DATA:         path stated in preprocessing

* main.py analyze [-h] [--file FILE] [--data DATA] [--name NAME]
    * -h, --help:   show this help message and exit
    * --file FILE:  use pickle file to analyze, otherwise file will be created
    * --data DATA:  path stated in preprocessing
    * --name NAME:  name of pickle file that will be created

* main.py preprocess [-h] [--data DATA] [--target TARGET]
                          [--slice_size SLICE_SIZE]
    * -h, --help:            show this help message and exit
    * --data DATA:           path of directory, that contain songs files
    * --target TARGET:       path for sliced files. There will be created two
                        directories "target_mid" and "target_wav"
    * --slice_size SLICE_SIZE:
                        slice size in seconds. Default is 0.25

* main.py predict [-h] --model {lstm,cnn} --file FILE [--data DATA]
    * -h, --help:          show this help message and exit
    * --model {lstm,cnn}
    * --file FILE:         file with trained model
    * --data DATA:         path stated in preprocessing






Work is based on two projects, which are avaliable [here ](https://medium.com/@alexissa122/generating-original-classical-music-with-an-lstm-neural-network-and-attention-abf03f9ddcb4) and [here ](https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5)
## License
[MIT](https://choosealicense.com/licenses/mit/)