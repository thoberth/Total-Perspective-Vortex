import matplotlib.pyplot as plt
from pyedflib import highlevel
import numpy as np
import os
import mne
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
	ls = os.listdir()[:60]
	for directory in tqdm(ls):
		if directory.startswith('S') and not directory.endswith('.txt'):
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
	os.chdir(PWD)
	return data


def parse_filter_data(data: dict):
	"""
	"""
	# focus on typical EEG Bands
	data = mne.io.concatenate_raws([data[key] for key in data])

	mne.datasets.eegbci.standardize(raw=data)

	montage = mne.channels.make_standard_montage("biosemi64")
	# print(montage.ch_names)
	data.set_montage(montage, on_missing='ignore')
	# data.plot()
	# plt.show()
	data.filter(l_freq=13, h_freq=30, verbose=False)
	# data.plot()
	# plt.show()
	events, event_id = mne.events_from_annotations(data, event_id =dict(T1=1, T2=2), verbose=False)
	data = mne.Epochs(data.filter(l_freq=13, h_freq=30, verbose=False),\
					events=events, event_id=event_id,\
					tmin=0.1, tmax=0.1, baseline=None, preload=True,\
					verbose=False)
	return data['T1'].get_data(copy=False), data['T2'].get_data(copy=False)


def streaming():
	pass


if __name__=='__main__':
	data = retrieve_data(PATH)
	data = parse_filter_data(data)
	print("Test")