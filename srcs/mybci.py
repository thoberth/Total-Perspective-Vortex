import mne
mne.set_log_level("CRITICAL")
import numpy as np
from tqdm.auto import tqdm
from sklearn.model_selection import cross_val_score, train_test_split
import argparse
from utils import retrieve_path, subject_argument, experiment_argument, action_argument, standardization_argument
import os
from training import train
# from predict import predict
# from cross_validation import cross_val



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="This program can train and predict")

	# argument of program
	# required
	parser.add_argument("-v", "--verbose")
	parser.add_argument("-p", "--plot", default=False, action="store_true", help="Plot the difference between raw data and data filtered")
	parser.add_argument("--standardization", default="mne", type=standardization_argument, help="The standardization method to use, should be 'mne' or 'zscore', default = 'mne'")
	parser.add_argument("action", nargs='?', default="cross_val", type=action_argument, help="Specify a number")
	parser.add_argument("-s", "--subject", nargs='*', type=int, default=[1, 109], help="The subject(s) to train or to predict")
	parser.add_argument("-e", "--experiment", nargs='*', type=int, default=[1, 14], help="The experiment(s) to train or to predict")

	# optional
	args = parser.parse_args()
	try:
		args.subject = subject_argument(args.subject)
		args.experiment = experiment_argument(args.experiment)
	except argparse.ArgumentTypeError as e:
		print(e)
		exit(1)
	path = retrieve_path()

	train(path, args.subject, args.experiment, args.standardization, args.plot)
