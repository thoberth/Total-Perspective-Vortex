import numpy as np
import argparse
import os
import json


def retrieve_path():
	with open('srcs/json/path.json', 'r') as f:
		data = json.load(f)
	return data['path_to_folder_corr'].replace("USERNAME", os.getenv("USER"))


def subject_argument(value): # ajouter la possibilite d'avoir plusieurs valeurs ou un range
	errorMsg = f"The subject value '{value}' must be an int between 1 and 109!"
	if isinstance(value, str):
		try :
			value = int(value)
		except:
			raise argparse.ArgumentTypeError(errorMsg)
		if not 1 < value < 110:
			raise argparse.ArgumentTypeError(errorMsg)
	elif isinstance(value, list):
		pass #here
	else:
		raise argparse.ArgumentTypeError(errorMsg)
	return value


def experiment_argument(value):
	errorMsg = f"The experiment value '{value}' must be an int between 1 and 14!"
	if isinstance(value, str):
		try :
			value = int(value)
		except:
			raise argparse.ArgumentTypeError(errorMsg)
		if not 1 < value < 110:
			raise argparse.ArgumentTypeError(errorMsg)
	elif isinstance(value, list):
		pass #here
	else:
		raise argparse.ArgumentTypeError(errorMsg)
	return value


def action_argument(value):
	if value not in ["predict", "train", "cross_val"]:
		raise argparse.ArgumentTypeError(f"The action value '{value}' must be 'train' or 'predict' or 'cross_val'")
	return value