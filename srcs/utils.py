import numpy as np
import argparse
import os
import json
import platform
import pickle
from sklearn.pipeline import Pipeline

def retrieve_path():
	with open('srcs/json/path.json', 'r') as f:
		data = json.load(f)
	if platform.system() == "Darwin":
		return os.getcwd() + "/data/"
	else:
		return data['path_to_folder_corr'].replace("USERNAME", os.getenv("USER"))


def subject_argument(value): # ajouter la possibilite d'avoir plusieurs valeurs ou un range
	errorMsg = f"The subject value '{value}' must be an int between 1 and 109!"
	if isinstance(value, list):
		if len(set(value)) != len(value):
			raise argparse.ArgumentTypeError(errorMsg)
		elif len(value) == 2:
			value.sort()
			if not (0 < value[0] < 110) or not (0 < value[1] < 110):
				raise argparse.ArgumentTypeError(errorMsg)
			return [i for i in range(value[0], value[1] + 1)]
		else:
			for val in value:
				if not 0 < val < 110:
					raise argparse.ArgumentTypeError(errorMsg)
			return value
	else:
		raise argparse.ArgumentTypeError(errorMsg)


def experiment_argument(value):
	errorMsg = f"The experiment value '{value}' must be at least an int between 1 and 14!"
	if isinstance(value, list):
		if len(set(value)) != len(value):
			raise argparse.ArgumentTypeError(errorMsg)
		elif len(value) == 2:
			value.sort()
			if not (0 < value[0] < 15) or not (0 < value[1] < 15):
				raise argparse.ArgumentTypeError(errorMsg)
			return [i for i in range(value[0], value[1] + 1)]
		else:
			for val in value:
				if not 0 < val < 15:
					raise argparse.ArgumentTypeError(errorMsg)
			return value
	else:
		raise argparse.ArgumentTypeError(errorMsg)

def action_argument(value):
	if value not in ["predict", "train", "cross_val"]:
		raise argparse.ArgumentTypeError(f"The action value '{value}' must be 'train' or 'predict' or 'cross_val'")
	return value
