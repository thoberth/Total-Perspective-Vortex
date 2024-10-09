from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, classification_report
import sys, os
from tqdm.auto import tqdm
from typing import List
from mne.io import read_raw_edf, concatenate_raws
from mne.datasets.eegbci import load_data, standardize
from mne import Epochs, events_from_annotations, annotations_from_events, set_log_level, pick_types
from mne.channels import make_standard_montage # "biosemi64"
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from itertools import chain
from termcolor import colored
import numpy as np
import matplotlib.pyplot as plt
from mne.decoding import CSP
from mne.preprocessing import ICA
from mne.datasets import eegbci
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
		events, event_id = events_from_annotations(data, event_id ='auto', verbose=False)
		mapping = {1: 'rest', 2: 'feet', 3: 'hands'}
		annot_from_events = annotations_from_events(
			events=events, event_desc=mapping, sfreq=data.info['sfreq'],
			orig_time=data.info['meas_date'])
		data.set_annotations(annot_from_events)
		montage = make_standard_montage("biosemi64")
		data.set_montage(montage, on_missing='ignore')
		eegbci.standardize(data)
		biosemi_montage = make_standard_montage('standard_1005')
		data.set_montage(biosemi_montage)
		if plot:
			data.compute_psd().plot()
			data.plot(scalings=dict(eeg=250e-6))
			plt.show()
		if data.info['sfreq'] != 160:
			print(colored(f"{file} cannot be used, the frequency is not valid", "red"))
			continue
		data.notch_filter(60, picks='eeg', fir_design="firwin", verbose=False)
		data.filter(l_freq=8, h_freq=30, fir_design='firwin', verbose=False)

		# if standardize_method == "mne":
		# 	standardize(raw=data)
		if plot:
			data.compute_psd().plot()
			data.plot(scalings=dict(eeg=250e-6))
			plt.show()
			plot=False
		data_clean.append(data)
	X = concatenate_raws(data_clean)
	# filter any bad channels that were identified to have artifacts
	picks = pick_types(X.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

	ica = ICA(n_components=15, method='fastica', random_state=97, max_iter='auto')
	ica.fit(X, picks=picks)
	if plot:
		ica.plot_components()
		plt.show()

	eog_channels = ['Fpz']  # Remplacez par les canaux appropriés si vous n'avez pas de canaux EOG explicites

	# Identifier les artefacts (par exemple, EOG pour les clignements d'yeux)
	eog_indices, eog_scores = ica.find_bads_eog(X, ch_name=eog_channels, threshold=1.5)
	ica.exclude = eog_indices  # Exclure les composantes identifiées

	# Reconstruire les signaux EEG sans les artefacts
	X = ica.apply(X.copy(), exclude=ica.exclude)
	X = Epochs(X, events=events, event_id=event_id, preload=True,\
					verbose=False)

	y = np.array(X.events[:, -1])
	X =	X.get_data()
	print(X.shape, y.shape)
	# if standardize_method == "zscore":
	X = z_score_normalize(X)
	return X, y


def train(path: str, subject: List, experiment: List, standardization: str, plot: bool):
	print("Downloading files if they aren't already downloaded...", set_log_level("CRITICAL", return_old_level=True))
	path_to_stored_data = []
	for index_subject in tqdm(subject):
		path_to_stored_data.append(load_data(subject=index_subject, runs=experiment, path=path))
	path_to_stored_data = list(chain(*path_to_stored_data))
	
	print("Retrieving data from edf files...")
	X, y = retrieve_raw_data(path_to_stored_data, standardization, plot)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
	print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
	pipe = Pipeline([('csp', CSP()), ('svc', SVC())], verbose=False)
	print("SVC", pipe.fit(X_train, y_train).score(X_test, y_test))
	# param_grid = {
	# 'csp__n_components': [16, 20, 24, 28, 32],  # Nombre de filtres spatiaux
	# 'csp__reg': [None,1, 0.2, 0.1, 0.01],  # Paramètres de régularisation
# }
	# grid = GridSearchCV(pipe, param_grid, cv=4)
	# grid.fit(X_train, y_train)

	# print("Meilleurs hyperparamètres :", grid.best_params_)
	# print("Meilleure précision sur le set d'entraînement :", grid.best_score_)

	# y_pred = grid.predict(X_test)
	# print("Rapport de classification :\n", classification_report(y_test, y_pred))
	# print("Précision sur le set de test :", accuracy_score(y_test, y_pred))


	# print(cross_val_score(pipe, X, y, cv=10))