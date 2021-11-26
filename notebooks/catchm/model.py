from sklearn.base import BaseEstimator, ClassifierMixin
from catchm.embeddings import InductiveDeepwalk

# SKLEARN CLASSIFIER
class CatchM(ClassifierMixin, BaseEstimator):
    """ An example classifier which implements a 1-NN algorithm.
    For more information regarding how to build your own classifier, read more
    in the :ref:`User Guide <user_guide>`.
    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.
    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """
    def __init__(self, dimensions, walk_len, walk_num, head_node_type = 'transfer', epochs=5, workers=1, window_size=5):
        self.dimensions = dimensions
        self.walk_len = walk_len
        self.walk_num = walk_num
        self.head_node_type = head_node_type
        self.epochs = epochs
        self.workers = workers
        self.window_size = window_size

    def fit(self, edgedict, y):
        """A reference implementation of a fitting function for a classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.
        Returns
        -------
        self : object
            Returns self.
        """

        # Create Inductive Deepwalk model
        self.idw = InductiveDeepwalk(dimensions=128, walk_len=20, walk_num=10, head_node_type = 'transfer', epochs=5, workers=4, window_size=5)
        self.idw.fit(edgedict)
        
        #-> XGBoost fit
        
        # Return the classifier
        return self

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        #-> embedding predict
        #-> Xgboost predict

        
        return 1


