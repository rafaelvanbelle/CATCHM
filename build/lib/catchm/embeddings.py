from networkx.readwrite import edgelist
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

import networkx as nx
from nodevectors import Node2Vec
import pandas as pd

from catchm.inductive import inductive_pooling
from catchm.network import create_network
from catchm.utils import EpochLogger

class InductiveDeepwalk(BaseEstimator, TransformerMixin):
    """ A template estimator to be used as a reference implementation.
    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.
    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    Examples
    --------
    >>> from skltemplate import TemplateEstimator
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> estimator = TemplateEstimator()
    >>> estimator.fit(X, y)
    TemplateEstimator()
    """
    def __init__(self, dimensions, walk_len, walk_num, epochs=5, workers=1, window_size=5, verbose=0):
        self.dimensions = dimensions
        self.walk_len = walk_len
        self.walk_num = walk_num
        self.epochs = epochs
        self.workers = workers
        self.window_size = window_size
        self.first_fit = True
        self.verbose = verbose

    def fit(self, X, y=None):
        """A reference implementation of a fitting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """

        if self.verbose > 0:
            print("Parsing input into network format.")

        self.G = create_network(X)
        transfer_nodes = nx.get_node_attributes(self.G, "type")
        transfers = [k for k,v in transfer_nodes.items() if v=='transfer' ]

        callbacks = []
        if self.verbose > 0:
            print("Running network representation algorithm.")
            epochlogger = EpochLogger()
            callbacks = [epochlogger]

        g2v = Node2Vec(
            n_components=self.dimensions,
            walklen = self.walk_len,
            epochs = self.walk_num,
            verbose = self.verbose,
            w2vparams={'workers': self.workers, 'window': self.window_size, 'callbacks': callbacks}
        )

        g2v.fit(self.G)
        self.model = g2v.model
        
        self.embeddings = np.zeros(shape=(len(transfers), self.dimensions))
       
            
        for id in transfers:
                self.embeddings[int(id), :] = self.model.wv[str(id)]
       
        
        self.is_fitted_ = True
        self.first_fit = True
        return self

    def transform(self, edgelist):
        """ A reference implementation of a predicting function.
        Parameters
        ----------
        G : Networkx graph containing test data
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        
        # Apply inductive mean pooling
        #X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        
        if self.first_fit:

            if self.verbose > 0:
                print("Retrieving embeddings for training data.")

            results = self.embeddings
            self.first_fit = False

        else:

            if self.verbose > 0:
                print("Running inductive pooling extension.")

            results = inductive_pooling(edgelist, embeddings=self.embeddings, G=self.G, workers=self.workers)
        
    
        return results

    