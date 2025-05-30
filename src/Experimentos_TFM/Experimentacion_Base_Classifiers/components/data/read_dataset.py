# import numpy as np
# from sktime.datasets import load_basic_motions


# def read_dataset(path: str) -> np.ndarray:
#     return load_basic_motions(return_X_y=True)


import numpy as np
from aeon.datasets import load_classification
from sklearn.preprocessing import LabelEncoder

# Lista de datasets multivariados de igual longitud que puedo descargar usando la API de AEON
multivariate_equal_length = [
    "ArticularyWordRecognition",
    "AtrialFibrillation",
    "BasicMotions",
    "Cricket",
    "DuckDuckGeese",
    "EigenWorms",
    "Epilepsy",
    "EthanolConcentration",
    "ERing",
    "FaceDetection",
    "FingerMovements",
    "HandMovementDirection",
    "Handwriting",
    "Heartbeat",
    "Libras",
    "LSST",
    "MotorImagery",
    "NATOPS",
    "PenDigits",
    "PEMS-SF",
    "PhonemeSpectra",
    "RacketSports",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "StandWalkJump",
    "UWaveGestureLibrary",
]


def read_datasets(
    name: str, split=None, extract_path=None, return_metadata=True, stats=False
) -> tuple:
    """Load a classification dataset.

    Loads a TSC dataset from extract_path, or from timeseriesclassification.com,
    if not on extract path.

    Data is assumed to be in the standard .ts format: each row is a (possibly
    multivariate) time series.
    Each dimension is separated by a colon, each value in a series is comma
    separated. For examples see aeon.datasets.data.tsc. ArrowHead is an example of
    a univariate equal length problem, BasicMotions an equal length multivariate
    problem.

    Data is stored in extract_path/name/name.ts, extract_path/name/name_TRAIN.ts and
    extract_path/name/name_TEST.ts.

    Parameters
    ----------
    name : str
        Name of data set. If a dataset that is listed in tsc_data_lists is given,
        this function will look in the extract_path first, and if it is not present,
        attempt to download the data from www.timeseriesclassification.com, saving it to
        the extract_path.
    split : None or str{"train", "test"}, default=None
        Whether to load the train or test partition of the problem. By default it
        loads both into a single dataset, otherwise it looks only for files of the
        format <name>_TRAIN.ts or <name>_TEST.ts.
    extract_path : str, default=None
        the path to look for the data. If no path is provided, the function
        looks in `aeon/datasets/data/`. If a path is given, it can be absolute,
        e.g. C:/Temp/ or relative, e.g. Temp/ or ./Temp/.
    return_metadata : boolean, default = True
        If True, returns a tuple (X, y, metadata)

    Returns
    -------
    X: np.ndarray or list of np.ndarray
    y: numpy array
        The class labels for each case in X
    metadata: optional
        returns the following meta data
        'problemname',timestamps, missing,univariate,equallength, class_values
        targetlabel should be false, and classlabel true

    Examples
    --------
    >>> from aeon.datasets import load_classification
    >>> X, y, meta = load_classification(name="ArrowHead") #DOCTEST +Skip
    """
    result = load_classification(  # noqa: N806
        name, split=split, extract_path=extract_path, return_metadata=return_metadata
    )

    if result is None:
        print(f"Could not load dataset '{name}'")
        # raise error
        raise ValueError(f"Dataset '{name}' not found.")

    if return_metadata:
        X, y, meta = result
    else:
        X, y = result

    if stats:
        le_ = LabelEncoder()
        y_transformed = le_.fit_transform(y)

        # mapping dictionary
        label_mapping = {i: label for i, label in enumerate(le_.classes_)}
        print(f"Label mapping dictionary: {label_mapping}")

        # Get the number of classes and their counts
        classes, counts = np.unique(y_transformed, return_counts=True)
        print(f"Classes: {classes}, Counts: {counts}")

        # Check for multivariate and get dimensions
        is_multivariate = X.ndim == 3
        if not is_multivariate and X.ndim == 2:
            print("Univariate dataset with 2D shape")
            print(f"X shape: {X.shape}")
            print("converting to 3D")

        n_samples, n_channels, n_timepoints = (
            X.shape if is_multivariate else (X.shape[0], 1, X.shape[1])
        )
        # Count missing values
        missing_values = np.isnan(X).sum()

        # Check for duplicates: it could be memory intensive
        # X_flattened = X.reshape(X.shape[0], -1)
        # unique_series = np.unique(X_flattened, axis=0)
        # duplicates = X.shape[0] - unique_series.shape[0]
        # Print the results
        print("split:", split)
        print(f"Shape of X: {X.shape}")
        print(f"Dataset: {name}")
        print(f"Samples: {n_samples}")
        print(f"Channels: {n_channels}")
        print(f"Timepoints: {n_timepoints}")
        print(f"Classes: {len(classes)}")
        print(f"Missing Values: {missing_values}")
        # print(f"Duplicates: {duplicates}")
    if return_metadata:
        return X, y, meta
    else:
        return X, y


# Example usage
if __name__ == "__main__":
    # Example usage
    dataset_name = "BasicMotions"
    X, y, meta = read_datasets(dataset_name, stats=True)
    print("metadata:")
    print(meta)
