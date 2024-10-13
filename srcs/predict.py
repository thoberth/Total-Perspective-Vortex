from typing import List
from training import retrieve_raw_data
from mne.datasets.eegbci import load_data
from utils import load_model
from itertools import chain

def predict(path: str, subject: List[str], experiment: List[str]):
	pipe=load_model()
	path_to_stored_data = []
	for index_subject in subject:
		path_to_stored_data.append(load_data(subject=index_subject, runs=experiment, path=path))
	path_to_stored_data = list(chain(*path_to_stored_data))
	X, y = retrieve_raw_data(path_to_stored_data, plot=False)

	print("\nPrediction score:", pipe.score(X, y))