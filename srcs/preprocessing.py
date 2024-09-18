import matplotlib.pyplot as plt
from pyedflib import highlevel
import numpy as np
import os
import mne
from sklearn.linear_model import SGDClassifier 
from tqdm.auto import tqdm
import sys
from typing import List


PWD=os.getcwd()
PATH='/mnt/nfs/homes/thoberth/sgoinfre/physionet.org/files/eegmmidb/1.0.0/'


def retrieve_data(path: str)-> dict:
	"""
	This function retrieve data from MNE file
	Returns:
		- a dictionnary name of file :  registration in MNE class
	"""
	if not os.path.exists(path):
		print(f"This path is incorrect {path} !", file=sys.stderr)
		exit(1)
	os.chdir(path)
	runs = [f'R{i:02}' for i in range(3, 15)]
	data = {}
	ls = os.listdir()[:40]
	for directory in tqdm(ls):
		try:
			os.chdir(directory)
			sub_ls=os.listdir()
			for file in sub_ls:
				file = file[:-4]
				if file[-3:] in runs:
					try:
						data[file] = mne.io.read_raw_edf(file+'.edf', verbose=False, preload=True)
						if data[file].info['sfreq'] != 160:
							del data[file]
							continue
					except:
						print(f"Something went wrong with file :{file}")
			os.chdir('..')
		except:
			pass
	os.chdir(PWD)
	return data


def parse_filter_data(data: dict, standardize : str = "mne"):
	"""
		This function receive extracted data from "mne.io.read_raw_edf",
		the function parse data, for each event and standardizes it.
		Args:
			data: a dict which contain name_of_experiment:data
			standardize: a str should be "mne" or "sklearn" it define
		the module uses to standardize the data
	"""
	if standardize not in ["sklearn", "mne"]:
		raise ValueError("Invalid value for standardize. It must be either 'mne' or 'sklearn'.")

	# focus on typical EEG Bands
	data = mne.io.concatenate_raws([data[key] for key in data])

	if standardize == 'mne':
		print("mne normalisation ...")
		mne.datasets.eegbci.standardize(raw=data)
	print(type(data.get_data()), data.get_data().shape)

	montage = mne.channels.make_standard_montage("biosemi64")
	data.set_montage(montage, on_missing='ignore')

	data.filter(l_freq=0.5, h_freq=30, verbose=False)

	events, event_id = mne.events_from_annotations(data, event_id = dict(T1=1, T2=2), verbose=False)
	data = mne.Epochs(data.filter(l_freq=13, h_freq=30, verbose=False),\
					events=events, event_id=event_id, preload=True,\
					verbose=False, baseline=None)
	X = data.get_data(copy=False)
	y = data.events[:, -1]#.reshape(-1, 1)
	return X, y


if __name__=='__main__':
	data = retrieve_data(PATH)
	data = parse_filter_data(data)
	print("Test")