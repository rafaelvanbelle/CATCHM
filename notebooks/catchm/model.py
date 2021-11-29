from sklearn.base import BaseEstimator, ClassifierMixin
from catchm.embeddings import InductiveDeepwalk
from sklearn.pipeline import Pipeline

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
    default_xgboost_params = {'eval_metric' = ['auc','aucpr', 'logloss'], 'n_estimators'=300, 'n_jobs'=8, 'learning_rate'=0.1, 'seed'=42, 'colsample_bytree' = 0.6, 'colsample_bylevel'=0.9, 'subsample' = 0.9}
    
    
    def __init__(self, dimensions, walk_len, walk_num, head_node_type = 'transfer', epochs=5, workers=1, window_size=5):
        self.dimensions = dimensions
        self.walk_len = walk_len
        self.walk_num = walk_num
        self.head_node_type = head_node_type
        self.epochs = epochs
        self.workers = workers
        self.window_size = window_size

    def fit(self, edgelist, y, xgboost_params={}):
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
        fit_embeddings(self, edgelist)
        
        #-> XGBoost fit
        fit_classifier(self, edgelist, y, xgboost_params)
        
        # Return the classifier
        return self

    def fit_embeddings(self, edgelist)

        self.embedder = InductiveDeepwalk(self.dimensions, self.walk_len, self.walk_num, self.head_node_type, self.epochs, self.workers, self.window_size)
        self.embedder.fit(edgelist)
    
    def fit_classifier(self, edgelist, y, xgboost_params=default_xgboost_params)

        self.classifier = xgb.XGBClassifier()
        pipe = Pipeline([('embedder', self.embedder), ('model', self.classifier)])
        pipe.fit(edgelist, y)


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
        self.embedder.transform()
        #-> Xgboost predict
        y_pred_proba = self.classifier.predict_proba()
        
        return 1


