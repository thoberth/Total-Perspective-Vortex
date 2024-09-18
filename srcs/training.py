from sklearn.pipeline import Pipeline
import sys, os
from tqdm.auto import tqdm
from typing import List
from mne.io import read_raw_edf, concatenate_raws
from mne.datasets.eegbci import load_data, standardize
from mne import Epochs, events_from_annotations
from mne.channels import make_standard_montage # "biosemi64"
from itertools import chain
from termcolor import colored
import numpy as np
from sklearn.preprocessing import StandardScaler

def retrieve_raw_data(path: str, standardize_method : str = "mne"):
	print("To avoid memory usage problem we are using data streaming...")
	data_clean = []
	for file in tqdm(path):
		data = read_raw_edf(file, verbose=False, preload=True)
		if data.info['sfreq'] != 160:
			print(colored(f"{file} cannot be used, the frequency is not valid", "red"))
			continue
		# data = concatenate_raws(data)
		if standardize_method == "mne":
			standardize(raw=data)
		# montage = make_standard_montage("biosemi64")
		# data.set_montage(montage, on_missing='ignore')
		data.filter(l_freq=0.5, h_freq=30, fir_design='firwin', verbose=False)
		events, event_id = events_from_annotations(data, event_id = dict(T1=1, T2=2), verbose=False)
		data = Epochs(data.filter(l_freq=13, h_freq=30, verbose=False),\
					events=events, event_id=event_id, preload=True,\
					verbose=False, baseline=None)
		data_clean.append(data)
	X = np.concatenate([i.get_data() for i in data_clean])
	if standardize_method == "sklearn":
		X = StandardScaler(X)
	y = np.concatenate([i.events[:, -1] for i in data_clean])
	return X, y

def train(path: str, subject: List, experiment: List):
	print("Downloading data if they aren't already downloaded...")
	path_to_stored_data = []
	for index_subject in tqdm(subject):
		path_to_stored_data.append(load_data(subject=index_subject, runs=experiment, path=path))
	path_to_stored_data = list(chain(*path_to_stored_data))
	
	print("Retrieving data from edf files...")
	X, y = retrieve_raw_data(path_to_stored_data)