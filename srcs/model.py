from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from myCSP import myCSP
import os
import pickle
from sklearn.model_selection import cross_val_score, train_test_split
from mne.decoding import CSP
from time import sleep
from sklearn.preprocessing import MinMaxScaler

class Model(BaseEstimator):
	def __init__(self, balanced: bool):
		super().__init__()
		if balanced:
			print("Creation d'un modele prenant en compte les donnees non-balances")
			self.model = Pipeline([('mycsp', myCSP(n_components=8)), ('svc', SVC(class_weight='balanced'))])
		else:
			self.model = Pipeline([('mycsp', myCSP(n_components=8)), ('svc', SVC(class_weight=None))])

	def fit(self, X, y):
		print("Fitting ...")
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
		cv_score = cross_val_score(self.model, X_train, y_train, cv=5)
		print(cv_score, '\nMean accuracy on train set: ', cv_score.mean())
		self.model.fit(X_train, y_train)
		print("Predict on Test set:")
		self.predict(X_test, y_test, verbose=False)
		return self

	def predict(self, X, y, verbose=True):
		counter = {"total":0, "class_1":0, "class_2":0}
		shape = X.shape
		total_class_1 = y[y==1].shape[0]
		total_class_2 = y[y==2].shape[0]
		for x, y_true in zip(X, y):
			y_pred = self.model.predict(x.reshape(1, shape[1], shape[2]))
			if verbose:
				print(f"\rTRUE VALUE IS [{y_true}], PREDICTED IS {y_pred}\t {y_true==y_pred}", end='')
				sleep(0.1)
			if y_true == y_pred:
				counter['total'] += 1
				if y_true == 1:
					counter['class_1'] += 1
				else:
					counter['class_2'] += 1
		print(f"\nMean accuracy = {counter['total'] / X.shape[0] * 100}")
		print(f"class_1 accuracy = ", counter['class_1'], "/", total_class_1)
		print(f"class_2 accuracy = ", counter['class_2'], "/", total_class_2)

	def score(self, X, y):
		return self.model.score(X, y)

	def save_model(self):
		path = os.getcwd()+'/model/'
		if not os.path.exists(path):
			os.makedirs('model')
		with open(path+'model.pickle', 'wb') as f:
			pickle.dump(self.model, f)

	def load_model(self) -> "Model":
		path=os.getcwd()+'/model/model.pickle'
		if not os.path.exists(path):
			raise FileNotFoundError()
		with open(path, 'rb') as f:
			self.model = pickle.load(f)
		return self