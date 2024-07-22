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

def retrieve_data(path: str)-> List[dict]:
	"""
	This function retrieve data from MNE file
	Returns:
		- a dictionnary name of file :  registration in MNE class
	"""
	if not os.path.exists(path):
		print(f"This path is incorrect {path} !", file=sys.stderr)
		exit(1)
	os.chdir(path)
	data = {}
	ls = os.listdir()
	for directory in ls:
		if directory.startswith('S') and not directory.endswith('.txt'):
			os.chdir(directory)
			sub_ls=os.listdir()
			for file in sub_ls:
				if file.endswith('.edf'):
					try:
						data[file[:-4]] = mne.io.read_raw_edf(file, verbose=False, preload=True)
					except:
						print(f"Something went wrong with this file :{file}")
			os.chdir('..')
	os.chdir(PWD)
	return data


def parse_filter_data(data: List[dict]):
	"""
	"""
	# focus on typical EEG Bands
	for key in data:
		data[key] = data[key].filter(l_freq=1, h_freq=30) 
	
	# Here sort data by "Label" / "Anotations"
	# For each person retrieve T1 and T2 for motion A and B



if __name__=='__main__':
	data = retrieve_data(PATH)
	data = parse_filter_data(data)
