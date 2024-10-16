
from typing import List
from mne.io import read_raw_edf, concatenate_raws
from mne.datasets.eegbci import load_data
from mne import Epochs, events_from_annotations, annotations_from_events, pick_types
from mne.channels import make_standard_montage
from itertools import chain
from termcolor import colored
import numpy as np
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from mne.datasets import eegbci
from model import Model
from tqdm.auto import tqdm

def z_score_normalize(eeg_data):
	mean_val = np.mean(eeg_data, axis=0)
	std_val = np.std(eeg_data, axis=0)
	return (eeg_data - mean_val) / std_val

def plot_function(raw):
		raw.plot(scalings=dict(eeg=250e-6))
		raw.compute_psd().plot()

		plt.show()

def apply_ica(raw, picks, plot):
	print("Compute ICA ...")
	ica = ICA(method='fastica', random_state=97, max_iter='auto')
	ica.fit(raw, picks=picks)

	eog_channels = ['Fpz']  # Remplacez par les canaux appropriés si vous n'avez pas de canaux EOG explicites

	# Identifier les artefacts (par exemple, EOG pour les clignements d'yeux)
	eog_indices, _ = ica.find_bads_eog(raw, ch_name=eog_channels, threshold=1.5)
	ica.exclude = eog_indices  # Exclure les composantes identifiées
	if plot:
		ica.plot_components(picks=ica.exclude, show=True)

	# Reconstruire les signaux EEG sans les artefacts
	print("Apply ICA ...")
	# print(f"Exclude : {ica.exclude}")
	raw = ica.apply(raw.copy(), exclude=ica.exclude)

	return raw


def retrieve_raw_data(raw_fnames: str, plot : bool = False, ica = False):
	raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
	events, event_id = events_from_annotations(raw, verbose=False)
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
		plot_function(raw)
	raw.notch_filter(60, picks='eeg', fir_design="firwin", verbose=False)
	raw.filter(l_freq=7, h_freq=30, fir_design='firwin', verbose=False)

	picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

	if ica:
		raw = apply_ica(raw, picks, plot)

	if plot:
		plot_function(raw)
	try:
		raw = Epochs(raw, events=events, event_id=dict(Feet=1, Hand=2), preload=True,\
						verbose=False, picks=picks, proj=True, tmin=-1.0, tmax=4.0)
	except ValueError:
		print(colored("\nThere is no data for training or prediction! Please select others files", "red"))
		exit(1)
	y = np.array(raw.events[:, -1])
	X =	raw.get_data()
	print("Standardize data ...")
	X = z_score_normalize(X)
	return X, y


def train(path: str, subject: List, experiment: List, plot : bool, ica : bool, balanced):
	print("Downloading files if they aren't already downloaded...")

	path_to_stored_data = []
	for index_subject in tqdm(subject):
		path_to_stored_data.append(load_data(subject=index_subject, runs=experiment, path=path))
	path_to_stored_data = list(chain(*path_to_stored_data))
	print("Retrieving data from edf files...")
	X, y = retrieve_raw_data(path_to_stored_data, plot, ica)

	model = Model(balanced).fit(X, y)

	model.save_model()