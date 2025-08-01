from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin
from warnings import warn

from ..utils import *

class BaseTimeClassifier(ClassifierMixin, BaseEstimator, metaclass=ABCMeta):

    def __init__(self, 
                 timestamps=None,
                 sampling_ratio=None, 
                 min_length=None):
        
        self.end2end = False
        
        self.timestamps = timestamps
        self.sampling_ratio = sampling_ratio

        self.min_length = min_length
        self.max_length = None

        self.classes_ = None
        self.class_prior = None

    def fit(self, X, y, cost_matrices=None):

        # check input integrity
        X, y = check_X_y(X, y)

        # CHECKING COST MATRICES INTEGRITY
        # ....
        # use the last dimension of X to get the max_length (valid for 2D or 3D)
        self.max_length = X.shape[-1]
        if self.min_length is None:
            warn("No min_length procided, using a minimum length of 1 by default")
            self.min_length = 1

        # Getting prior probabilities
        self.classes_ = np.unique(y)
        self.class_prior = np.array([np.sum(y == class_) / len(y) for class_ in self.classes_])

        # check timestamps parameters validity as well as sampling ratio
        if self.timestamps is not None:
        
            self.timestamps = check_timestamps(self.timestamps)
            self.nb_classifiers = len(self.timestamps)
            if self.sampling_ratio is not None:
                warn("Both 'timestamps' and 'sampling_ratio' are defined, in that case" 
                     "argument 'sampling_ratio' is ignored")
                self.sampling_ratio = None

        elif self.sampling_ratio is not None:
            if not isinstance(self.sampling_ratio, float) \
                    and not isinstance(self.sampling_ratio, int):
                raise TypeError(
                    "Argument 'sampling_ratio' should be a strictly positive float between 0 and 1.")
            if self.sampling_ratio <= 0 or self.sampling_ratio > 1:
                raise ValueError(
                    "Argument 'sampling_ratio' should be a strictly positive float between 0 and 1.")
            
            self.nb_classifiers = np.minimum(int(1/self.sampling_ratio), self.max_length - self.min_length + 1)
            
        else:
            warn("No 'sampling_ratio' or pre-defined list of 'timestamps' "
                 "provided, using default 5'%' sampling, i.e. 20 classifiers")
            self.nb_classifiers = 20
            
        if self.timestamps is None:
            self.timestamps = np.array(list(set(
                [int((self.max_length - self.min_length) * i / self.nb_classifiers) + self.min_length
                 for i in range(1, self.nb_classifiers+1)]
            )))
            # update nb_classifiers if the previously setted value
            # was too large for example
            self.nb_classifiers = len(self.timestamps)

        # sort to avoid mismatch 
        self.timestamps = np.sort(self.timestamps)
            
        self._fit(X, y, cost_matrices)

        return self
    
    def predict_proba(self, X, cost_matrices=None):

        # check input integrity, not all ts have to be same length
        X, _ = check_X_y(X, None, equal_length=False)
        # Group X by batch of same length
        grouped_X = self._grouped_by_length(X)

        return self._predict_proba(grouped_X, cost_matrices)
    
    def predict_past_proba(self, X, cost_matrices=None):
        
        # check input integrity, not all ts have to be same length
        X, _ = check_X_y(X, None, equal_length=False)
        # Group X by batch of same length
        grouped_X = self._grouped_by_length(X)

        return self._predict_past_proba(grouped_X, cost_matrices)
    
    def predict(self, X, cost_matrices=None):
        """
        Predict class labels for samples in X.
    
        Parameters
        ----------
            X : array-like, shape (n_samples, n_timestamps)
                The input time series, potentially of various size.
            cost_matrices : object, optional (default=None)
                The input cost matrices, could be used for cost-sensitive learning.
    
        Returns
        -------
            y : ndarray, shape (n_samples)
                The labels for each sample the corresponding classifier has predicted.
        """
        try:
            probas = self.predict_proba(X, cost_matrices)
            return probas.argmax(axis=-1)
        except (NotImplementedError, AttributeError):
            # Group X by batch of same length
            X, _ = check_X_y(X, None, equal_length=False)
            grouped_X = self._grouped_by_length(X)
            # Use direct predict method instead
            return self._predict(grouped_X, cost_matrices)
    
    def _grouped_by_length(self, X):

        """
        Group time series in X by their number of timestamps (last dimension).
        
        For each series in X (assumed to be in shape (D, T)), if its time length (T)
        is not present in self.timestamps and is greater than the smallest valid timestamp,
        the series is truncated to the nearest valid timestamp (i.e. the largest value
        in self.timestamps that is less than or equal to the series’ length).
        
        Parameters
        ----------
        X : iterable of numpy.ndarray
            A collection of time series, each with shape (D, T), where D is the number of dimensions
            and T is the number of timestamps.
        
        Returns
        -------
        grouped_X : dict
            A dictionary where the keys are valid timestamps (lengths) and the values are lists of time series
            that have been (if necessary) truncated to that length.
        
        Side Effects
        ------------
        if a series is shorter than self.timestamps[0], it doesn't trigger the truncation block and will be 
        grouped under its actual length as its key in the dictionary.
        A warning is issued if any series were truncated because their original length did not match any of the
        fitted timestamps.
        """

        truncated = False
        grouped_X = {}
        for serie in X:
            length = serie.shape[-1]
            if length not in self.timestamps and \
                length > self.timestamps[0]:
                # truncate to nearest valid timestamp
                filtered = filter(lambda x: x <= length, self.timestamps)
                length = min(filtered, key=lambda x: length-x, default=None)
                if length is not None:
                    serie = serie[..., :length]
                    truncated = True

            if length in grouped_X.keys():
                grouped_X[length].append(serie)
            else:
                grouped_X[length] = [serie]
        
        if truncated:
            warn("Some time series were truncated during prediction since no classifier was fitted for their lengths.")
        
        return grouped_X
    
    @abstractmethod
    def _fit(self, X, y, cost_matrices):
        """Fit the classifier(s) to 
        target y
        """
    
    @abstractmethod
    def _predict_proba(self, grouped_X, cost_matrices):
        """Predict probabilities for each 
        class to be true label
        """
    
    @abstractmethod
    def _predict_past_proba(self, grouped_X, cost_matrices):
        """Predict probabilities for each 
        class to be true label, for each 
        past timestamps 
        """
    
    @abstractmethod
    def _predict(self, grouped_X, cost_matrices):
        """Predict the class labels directly
        (used when predict_proba is not available)
        """
        pass