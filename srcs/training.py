from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
import sys, os
from tqdm.auto import tqdm
from typing import List
from mne.io import read_raw_edf, concatenate_raws
from mne.datasets.eegbci import load_data, standardize
from mne import Epochs, events_from_annotations
from mne.channels import make_standard_montage # "biosemi64"
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.decomposition import PCA
from itertools import chain
from termcolor import colored
import numpy as np
import matplotlib.pyplot as plt
from mne.decoding import CSP
from multiprocessing import cpu_count

def z_score_normalize(eeg_data):
	mean_val = np.mean(eeg_data, axis=0)
	std_val = np.std(eeg_data, axis=0)
	return (eeg_data - mean_val) / std_val

def retrieve_raw_data(path: str, standardize_method : str = "mne", plot : bool = False ):
	#TO DO factoriser la fonction et plot une seule windows avec tout les avant/apres
	data_clean = []
	for i, file in enumerate(tqdm(path)):
		data = read_raw_edf(file, verbose=False, preload=True)
		montage = make_standard_montage("standard_1020")
		data.set_montage(montage, on_missing='ignore')
		if plot:
			data.compute_psd().plot()
			data.plot(scalings=dict(eeg=250e-6))
			plt.show()
		if data.info['sfreq'] != 160:
			print(colored(f"{file} cannot be used, the frequency is not valid", "red"))
			continue
		data.notch_filter(60, picks='eeg', method="iir", verbose=False)
		data.filter(l_freq=8, h_freq=20, fir_design='firwin', verbose=False)
		if standardize_method == "mne":
			standardize(raw=data)
		if plot:
			data.compute_psd().plot()
			data.plot(scalings=dict(eeg=250e-6))
			plt.show()
			plot=False
		events, event_id = events_from_annotations(data, event_id ='auto', verbose=False)
		# print(event_id)
		data = Epochs(data, events=events, event_id=event_id, preload=True,\
					verbose=False)
		# print(data.get_data())
		# break
		# print(file, data.get_data().shape)
		# if i> 14:
		# 	exit()
		data_clean.append(data)
	X = np.concatenate([i.get_data() for i in data_clean])
	y = np.concatenate([i.events[:, -1] for i in data_clean])
	if standardize_method == "zscore":
		X = z_score_normalize(X)
	return X, y


def train(path: str, subject: List, experiment: List, standardization: str, plot: bool):
	print("Downloading files if they aren't already downloaded...")
	path_to_stored_data = []
	for index_subject in tqdm(subject):
		path_to_stored_data.append(load_data(subject=index_subject, runs=experiment, path=path))
	path_to_stored_data = list(chain(*path_to_stored_data))
	
	print("Retrieving data from edf files...")
	X, y = retrieve_raw_data(path_to_stored_data, standardization, plot)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
	print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
	pipe = Pipeline([('csp', CSP(n_components=15)), ('lda', LDA())], verbose=False)
	print(cross_val_score(pipe, X, y, cv=10))