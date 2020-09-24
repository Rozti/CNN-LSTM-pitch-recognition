import argparse
import os

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='types of operation', dest='command')
train_parser = subparsers.add_parser("train")
train_parser.add_argument("--model", required=True, choices=["lstm", "cnn"])
train_parser.add_argument("--data", default="train/train_quad", help = 'path stated in preprocessing')

analyze_parser = subparsers.add_parser("analyze")
analyze_parser.add_argument("--file", help = "use pickle file to analyze, otherwise file will be created")
analyze_parser.add_argument("--data",  help = 'path stated in preprocessing')
analyze_parser.add_argument("--name", help = 'name of pickle file that will be created', default='data.pickle')

preprocess_parser = subparsers.add_parser("preprocess")
preprocess_parser.add_argument("--data", help = 'path of directory, that contain songs files', default="/home/andrzej/PycharmProjects/MGR/Magisterka/www.audiolabs-erlangen.de/content/resources/MIR/SMD/02-midi/data")
preprocess_parser.add_argument("--target", default='train/train_quad', help = 'path for sliced files. There will be created two directories "target_mid" and "target_wav"')
preprocess_parser.add_argument("--slice_size", default=0.25, type= float, help = 'slice size in seconds. Default is 0.25')

predict_parser = subparsers.add_parser("predict")
predict_parser.add_argument("--model", required=True, choices=["lstm", "cnn"])
predict_parser.add_argument("--file", required=True, help="file with trained model")
predict_parser.add_argument("--data", default="train/train_quad", help = 'path stated in preprocessing')


args = parser.parse_args()
print (args.command)
if(args.command == "train"):
    import lstm2 as train
    data = args.data
    if data.endswith('/'):
        data = data[:-1]
    if args.model == "cnn":
        train.start_cnn(data)
    else:
        train.start(data)

if(args.command == "analyze"):
    import data_analysis as da
    print(args)
    if args.file != None:
        if (args.data != None):
            print("file was stated, data argument will be omitted")
        if (args.name != None):
            print("file was stated, name argument will be omitted")
        da.analyse_songs(args.file)
    elif args.data != None:
        data = args.data
        if data.endswith('/'):
            data = data[:-1]
        data = da.get_notes(mid_dir=data + "_mid", png_dir=data + "_wav")

        da.save(args.name, data)
        da.analyse_songs(args.name)
    else:
        print("both path and file were not provided. Terminating process. ")

if(args.command == "preprocess"):
    target = args.target
    directory = args.data
    import playground as preprocess
    target_segment_len = args.slice_size

    for filename in os.listdir(directory):
        if filename.endswith(".mid"):
            midfile = os.path.join(directory, filename)
            print(midfile)
            preprocess.split_midi(midfile, target, target_segment_len=target_segment_len)

if(args.command == "predict"):
    import merge_predict2 as predict

    data = args.data
    if data.endswith('/'):
        data = data[:-1]
    if(args.model == "cnn"):
        predict.cnn_predict(mid_dir=data + "_mid", png_dir=data + "_wav", filepath=args.file)
    else:
        predict.merged_predict(mid_dir=data + "_mid", png_dir=data + "_wav", filepath=args.file)
