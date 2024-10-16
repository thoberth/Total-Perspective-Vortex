from typing import List
from training import retrieve_raw_data
from mne.datasets.eegbci import load_data
from itertools import chain
from model import Model
from tqdm.auto import tqdm

def predict(path: str, subject: List[str], experiment: List[str]):
	print("Recuperation du model ...")
	model = Model(balanced=False).load_model()
	path_to_stored_data = []
	print("Downloading files if they aren't already downloaded...")
	for index_subject in tqdm(subject):
		path_to_stored_data.append(load_data(subject=index_subject, runs=experiment, path=path))
	path_to_stored_data = list(chain(*path_to_stored_data))
	X, y = retrieve_raw_data(path_to_stored_data, plot=False)
	model.predict(X, y)