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

def retrieve_raw_data(raw_fnames: str, standardize_method : str = "mne", plot : bool = False ):
	#TO DO factoriser la fonction et plot une seule windows avec tout les avant/apres

	raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
	events, event_id = events_from_annotations(raw, event_id='auto', verbose=False)
	mapping = {1: 'rest', 2: 'feet', 3: 'hands'}
	annot_from_events = annotations_from_events(
		events=events, event_desc=mapping, sfreq=raw.info['sfreq'],
		orig_time=raw.info['meas_date'])
	raw.set_annotations(annot_from_events)
	eegbci.standardize(raw)
	biosemi_montage = make_standard_montage('standard_1005')
	raw.set_montage(biosemi_montage)
	raw.set_eeg_reference(projection=True)

	if plot:
		pass
		#create a fonction to plot raw_data before and after
	# 		data.compute_psd().plot()
	# 		data.plot(scalings=dict(eeg=250e-6))
	# 		plt.show()

	raw.notch_filter(60, picks='eeg', fir_design="firwin", verbose=False)
	raw.filter(l_freq=7, h_freq=30, fir_design='firwin', verbose=False)


	picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

	print("Compute ICA ...")
	ica = ICA(method='fastica', random_state=97, max_iter='auto')
	ica.fit(raw, picks=picks)
	if plot:
		ica.plot_components()
		plt.show()

	eog_channels = ['Fpz']  # Remplacez par les canaux appropriés si vous n'avez pas de canaux EOG explicites

	# Identifier les artefacts (par exemple, EOG pour les clignements d'yeux)
	eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=eog_channels, threshold=1.5)
	ica.exclude = eog_indices  # Exclure les composantes identifiées

	# Reconstruire les signaux EEG sans les artefacts
	print("Apply ICA ...")
	print(f"Exclude : {ica.exclude}")
	raw = ica.apply(raw.copy(), exclude=ica.exclude)

	raw = Epochs(raw, events=events, event_id=event_id, preload=True,\
					verbose=False, picks=picks, proj=True)

	y = np.array(raw.events[:, -1])
	X =	raw.get_data()
	print("Standardize data ...")
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

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
	print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
	pipe = Pipeline([('csp', CSP(n_components=6)), ('svc', SVC())], verbose=False)

	cv_score = cross_val_score(pipe, X_train, y_train, cv=5)
	print(cv_score, cv_score.mean(), sep='\nMean : ')
	print("Test: ",  pipe.fit(X_train, y_train).score(X_test, y_test))

# 	param_grid = {
# 	'csp__n_components': [5, 6, 7, 8],  # Nombre de filtres spatiaux
# 	'csp__reg': [None, 1, 0.1, 0.01],  # Paramètres de régularisation
# }
# 	grid = GridSearchCV(pipe, param_grid, cv=5)
# 	grid.fit(X_train, y_train)

# 	print("Meilleurs hyperparamètres :", grid.best_params_)
# 	print("Meilleure précision sur le set d'entraînement :", grid.best_score_)

# 	y_pred = grid.predict(X_test)
# 	print("Rapport de classification :\n", classification_report(y_test, y_pred))
# 	print("Précision sur le set de test :", accuracy_score(y_test, y_pred))