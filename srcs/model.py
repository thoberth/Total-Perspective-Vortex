from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from myCSP import myCSP
import os
import pickle
from sklearn.model_selection import cross_val_score, train_test_split
from mne.decoding import CSP
from time import sleep

class Model(BaseEstimator):
	def __init__(self):
		super().__init__()
		self.model = Pipeline([('mycsp', myCSP()), ('svc', SVC())])

	def fit(self, X, y):
		print("\rFitting ..."+ " "*20)
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
		cv_score = cross_val_score(self.model, X_train, y_train, cv=5)
		print(cv_score, '\nMean accuracy on train set: ', cv_score.mean())
		self.model.fit(X_train, y_train)
		print("Predict on Test set:")
		self.predict(X_test, y_test, verbose=False)
		return self

	def predict(self, X, y, verbose=True):
		counter = 0
		shape = X.shape
		for x, y_true in zip(X, y):
			y_pred = self.model.predict(x.reshape(1, shape[1], shape[2]))
			if verbose:
				print(f"\nTRUE VALUE IS [{y_true}], PREDICTED IS {y_pred}\t {y_true==y_pred}")
				sleep(0.2)
			if y_true == y_pred:
				counter += 1
		print(f"Mean accuracy = {counter / X.shape[0] * 100}")
		

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