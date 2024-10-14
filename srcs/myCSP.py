import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class myCSP(BaseEstimator, TransformerMixin):
	def __init__(self, n_components=4):
		"""
		Initialise la classe CSP.
		- n_components: Nombre de filtres spatiaux à conserver (doit être pair).
		"""
		self.n_components = n_components

	def fit(self, X, y):
		"""
		Ajuste le modèle CSP aux données.
		- X: ndarray de forme (n_trials, n_channels, n_samples)
		- y: ndarray de forme (n_trials,), étiquettes des classes (0 ou 1)
		"""
		# Vérification du nombre de classes
		classes = np.unique(y)
		if len(classes) != 2:
			raise ValueError("CSP ne supporte que deux classes.")

		# Séparation des données par classe
		X1 = X[y == classes[0]]
		X2 = X[y == classes[1]]

		# Calcul des matrices de covariance moyennes
		cov1 = self._compute_covariance_matrix(X1)
		cov2 = self._compute_covariance_matrix(X2)

		# Matrice de covariance composite
		cov_combined = cov1 + cov2

		# Décomposition en valeurs propres
		eigenvalues, eigenvectors = np.linalg.eigh(cov_combined)

		# Tri décroissant des valeurs propres
		idx = np.argsort(eigenvalues)[::-1]
		eigenvectors = eigenvectors[:, idx]
		eigenvalues = eigenvalues[idx]

		# Calcul de la matrice de blanchiment
		whitening_matrix = np.diag(1.0 / np.sqrt(eigenvalues)).dot(eigenvectors.T)

		# Transformation des matrices de covariance
		S1 = whitening_matrix.dot(cov1).dot(whitening_matrix.T)

		# Décomposition conjointe
		eigenvalues_s1, eigenvectors_s1 = np.linalg.eigh(S1)

		# Calcul des filtres CSP
		projection = eigenvectors_s1.T.dot(whitening_matrix)

		# Sélection des filtres
		n_filters = self.n_components // 2
		filters = np.vstack([projection[:n_filters, :], projection[-n_filters:, :]])
		self.filters_ = filters

		return self

	def transform(self, X):
		"""
		Applique la transformation CSP aux données.
		- X: ndarray de forme (n_trials, n_channels, n_samples)
		"""
		X_transformed = []
		for trial in X:
			# Projection du signal
			Z = self.filters_.dot(trial)
			# Calcul de la variance normalisée
			var = np.var(Z, axis=1)
			var /= var.sum()
			# Logarithme des variances
			X_transformed.append(np.log(var))
		return np.array(X_transformed)

	def _compute_covariance_matrix(self, X):
		"""
		Calcule la matrice de covariance moyenne normalisée.
		- X: ndarray de forme (n_trials, n_channels, n_samples)
		"""
		cov_matrices = []
		for trial in X:
			cov = np.cov(trial)
			cov /= np.trace(cov)
			cov_matrices.append(cov)
		return np.mean(cov_matrices, axis=0)