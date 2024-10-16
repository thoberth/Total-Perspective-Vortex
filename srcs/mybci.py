import mne
mne.set_log_level("CRITICAL")
import numpy as np
from tqdm.auto import tqdm
from sklearn.model_selection import cross_val_score, train_test_split
import argparse
from utils import retrieve_path, subject_argument, experiment_argument, action_argument
import os
from training import train
from predict import predict



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="This program can train and predict")

	parser.add_argument("-ica", default=False, action="store_true", help="Use ICA of MNE module for artifact removal")
	parser.add_argument("-balanced", default=False, action="store_true", help="Train the model as a balanced dataset")
	parser.add_argument("-p", "--plot", default=False, action="store_true", help="Plot the difference between raw data and data filtered")
	parser.add_argument("action", nargs='?', default="cross_val", type=action_argument, help="Specify a number")
	parser.add_argument("-s", "--subject", nargs='*', type=int, default=[1, 21], help="The subject(s) to train or to predict")
	parser.add_argument("-e", "--experiment", nargs='*', type=int, default=[3, 14], help="The experiment(s) to train or to predict")

	args = parser.parse_args()
	try:
		args.subject = subject_argument(args.subject)
		args.experiment = experiment_argument(args.experiment)
	except argparse.ArgumentTypeError as e:
		print(e)
		exit(1)
	path = retrieve_path()

	if args.action == 'train':
		train(path, args.subject, args.experiment, args.plot, args.ica, args.balanced)
	elif args.action == 'predict':
		predict(path, args.subject, args.experiment)
