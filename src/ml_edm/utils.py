import numpy as np
import pandas as pd

from sklearn.utils.multiclass import check_classification_targets
from warnings import warn 

def check_X_y(X, y, equal_length=True):
    """
    Check X and y and return them as NumPy arrays suitable for multivariate time series.

    The intended format for X is (N, D, T):
      - N: number of samples,
      - D: number of dimensions (features),
      - T: number of timestamps.
      
    For univariate data provided as 1D arrays (each sample of shape (T,)), the function converts the
    list to a 2D array (N, T) and then adds a singleton dimension to produce (N, 1, T). If equal_length is
    False, time series (either univariate or multivariate) that are of varying lengths are padded 
    along the time axis (the T dimension) with NaNs.

    Parameters:
        X: list, DataFrame, or np.ndarray
            Time series dataset. Can be a list of time series (each sample as a 1D or 2D array),
            a DataFrame, or a NumPy array.
        y: list or np.ndarray
            Labels corresponding to the examples in X.
        equal_length: bool, default=True
            Whether all time series are expected to have the same length.

    Returns:
        X: np.ndarray
            Array of shape (N, D, T).
        y: np.ndarray
            Array of labels.
    """
    # Process X if it is a list.
    if isinstance(X, list):
        sample0 = np.array(X[0])
        if sample0.ndim == 1:
            # Univariate data: each sample is 1D with shape (T,).
            if not equal_length:
                # Determine the maximum length among samples.
                max_length = max(len(x) for x in X)
                padded = []
                for ts in X:
                    ts = np.array(ts)
                    T = ts.shape[0]
                    if T < max_length:
                        ts = np.pad(ts, (0, max_length - T), mode='constant', constant_values=np.nan)
                    padded.append(ts)
                X = np.array(padded)  # Now X has shape (N, max_length)
            else:
                X = np.array(X)  # (N, T) where T is already equal for all samples.
        elif sample0.ndim == 2:
            # Multivariate data: each sample is 2D with shape (D, T).
            if not equal_length:
                # Determine maximum time length among samples.
                max_length = max(x.shape[1] for x in [np.atleast_2d(x) for x in X])
                padded = []
                for x in X:
                    arr = np.atleast_2d(x)  # Ensure shape is (D, T)
                    T = arr.shape[1]
                    if T < max_length:
                        pad_width = max_length - T
                        # Pad only along the time axis (axis 1).
                        arr = np.pad(arr, ((0, 0), (0, pad_width)), mode='constant', constant_values=np.nan)
                    padded.append(arr)
                X = np.array(padded)  # Now X has shape (N, D, max_length)
            else:
                X = np.stack([np.atleast_2d(x) for x in X])
        else:
            X = np.array(X)
    elif isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    elif not isinstance(X, np.ndarray):
        raise TypeError("X should be a list, DataFrame, or NumPy array (2D or 3D) of time series.")

    # At this point, X should be a NumPy array.
    # Allow arrays in 2D or 3D.
    if X.ndim not in [2, 3]:
        raise ValueError("X should be a 2D or 3D array.")
    
    # If X is 2D (univariate with shape (N, T)), convert to (N, 1, T).
    if X.ndim == 2:
        X = X[:, np.newaxis, :]

    # Validate equal length along the time dimension (axis 2) for (N, D, T).
    if equal_length:
        T = X[0].shape[1]  # The expected number of timestamps.
        for sample in X:
            if sample.shape[1] != T:
                raise ValueError("All time series must have the same number of timestamps (axis 2 in each sample).")
    
    if len(X) == 0:
        raise ValueError("Dataset X is empty.")

    # Process y
    if y is not None:
        if isinstance(y, list):
            y = np.array(y)
        elif not isinstance(y, np.ndarray):
            raise TypeError("y should be a list or NumPy array of labels.")
        if len(y) != X.shape[0]:
            raise ValueError("The size of y must match the number of examples in X.")
        check_classification_targets(y)

    return X, y

def check_X_probas(X_probas):

    # X_probas
    if isinstance(X_probas, list):
        X_probas = np.array(X_probas)
    elif isinstance(X_probas, pd.DataFrame):
        X_probas = X_probas.to_numpy()
    elif not isinstance(X_probas, np.ndarray):
        raise TypeError(
            "X_probas should be a 2-dimensional list, array or DataFrame of size (N, K) with N the number "
            "of examples and K the number of classes probabilities.")
    
    if X_probas.ndim != 2:
        raise ValueError(
            "X_probas should be a 2-dimensional list, array or DataFrame of size (N, K) with N the number "
            "of examples and K the number of classes probabilities.")
    
    if len(X_probas) == 0:
        raise ValueError("Dataset 'X_probas' to predict triggering on is empty.")

    return X_probas

def check_X_past_probas(X_past_probas):

    # X_probas
    if isinstance(X_past_probas, list):
        X_past_probas = np.array(X_past_probas)
    elif isinstance(X_past_probas, pd.DataFrame):
        X_past_probas = X_past_probas.to_numpy()
    elif not isinstance(X_past_probas, np.ndarray):
        raise TypeError(
            "X_past_probas should be a 3-dimensional list, array or DataFrame of size (T, N, K) with T the number of timepoints,"
            "N the number of examples and K the number of classes probabilities.")
    
    if X_past_probas.ndim != 3:
        raise ValueError(
            "X_past_probas should be a 3-dimensional list, array or DataFrame of size (T, N, K) with T the number of timepoints,"
            " N the number of examples and K the number of classes probabilities.")
    
    if X_past_probas.shape[1] == 0:
        raise ValueError("Dataset 'X_past_probas' to predict triggering on is empty.")

    return X_past_probas

def check_timestamps(timestamps):

    if isinstance(timestamps, list):
        timestamps = np.array(timestamps)
    elif not isinstance(timestamps, np.ndarray):
        raise TypeError("Argument 'timestamps' should be a list or array of positive int.")
    if len(timestamps) == 0:
            raise ValueError("List argument 'timestamps' is empty.")
    for t in timestamps:
        if not (isinstance(t, np.int32) or isinstance(t, np.int64)):
            raise TypeError("Argument 'timestamps' should be a list or array of positive int.")
        if t < 0:
            raise ValueError("Argument 'timestamps' should be a list or array of positive int.")
                
    if len(np.unique(timestamps)) != len(timestamps):
        timestamps = np.unique(timestamps)
        warn("Removed duplicates in argument 'timestamps'.")
    
    if 0 in timestamps:
        timestamps = np.nonzero(timestamps)[0]
        warn("Removed 0 from 'timestamps', first valid timestamps is usually 1.")

    return timestamps