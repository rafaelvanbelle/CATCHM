from sklearn.base import BaseEstimator, ClassifierMixin
from catchm.embeddings import InductiveDeepwalk
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
import numpy as np
import xgboost as xgb



# SKLEARN CLASSIFIER
class Catchm(ClassifierMixin, BaseEstimator):
    """

    Implementation of the CATCHM approach. This class contains two essential parts: 
    1) a network representation learning algorithm (based on Deepwalk), 2) a classifier (XGBoost).

    Parameters
    ----------
    dimensions : int, default=124
        Number of dimensions for the network vectors.

    walk_len : int, default=
        Length of a single random walk in the network. 

    walk_num : int, default= 
        Number of walks for a single node in the network. 

    window_size : int, default=5
        Maximum distance between the current and predicted word within a random walk.

    epochs : int, default=5
        Number of iterations (epochs) over the corpus of random walks. 

    workers : int, default=1
        Number of CPU cores used when parallelizing over classes.

    xgboost_params : dict, default={}
        Dict of parameters to pass to the XGBClassifier class. See XGBoost package documentation for more information.
    """
    
    
    def __init__(self, dimensions=124, walk_len=10, walk_num=20, epochs=5, workers=1, window_size=5, xgboost_params=None, check_input=True, verbose=0):
        
        self.dimensions = dimensions
        self.walk_len = walk_len
        self.walk_num = walk_num
        
        self.epochs = epochs
        self.workers = workers
        self.window_size = window_size
        self.xgboost_params = xgboost_params
        self.check_input = check_input
        self.verbose = verbose

    def fit(self, X, y):
        """
        ----------
        X : array-like, shape (n_samples, 2)
            The training input edgelist.
        y : array-like, shape (n_samples,)
            The target values. An array of int.
        Returns
        -------
        self : object
            Returns self.
        """

        if self.verbose > 0:
            print("Creating network representation model.")
        
        self.embedder = InductiveDeepwalk(self.dimensions, self.walk_len, self.walk_num, self.epochs, self.workers, self.window_size, self.verbose)
        
        if self.verbose > 0:
            print("Finished creating network representation model.")
            print("Training pipeline (embeddings + classifier)")
        
        self.classifier = xgb.XGBClassifier()
        self.pipe = Pipeline([('embedder', self.embedder), ('model', self.classifier)])
        self.pipe.fit(X, y)
        
        self.is_fitted_ = True
        # Return the classifier
        return self

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, 2)
            The input edgelist.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """

        check_is_fitted(self, 'is_fitted_')
        y_pred = self.pipe.predict(X)
        
        return y_pred


    def predict_proba(self, X):
        """ A reference implementation of a prediction for a classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, 2)
            The input edgelist.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            The score for each sample.
        """

        check_is_fitted(self, 'is_fitted_')
        y_pred_proba = self.pipe.predict_proba(X)
        
        return y_pred_proba


