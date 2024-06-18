import pyedflib
import numpy as np
import os
PATH='/mnt/nfs/homes/thoberth/sgoinfre/physionet.org/files/eegmmidb/1.0.0/'

if __name__=='__main__':
	os.chdir(PATH)
	ls = os.listdir()
	for direc in ls:
		if direc.startswith('S') and not (direc.endswith('.txt')):
			print(direc)