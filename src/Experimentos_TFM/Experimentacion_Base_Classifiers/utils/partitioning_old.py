### EXAMPLE USAGE ###
import json
import os

import numpy as np

# # https://github.com/aeon-toolkit/aeon/blob/v0.4.0/aeon/forecasting/model_selection/_split.py
# from aeon.forecasting.model_selection import (
#     # CutoffSplitter,
#     ExpandingWindowSplitter,
#     # SingleWindowSplitter,
#     # SlidingWindowSplitter,
#     # temporal_train_test_split,
# )
from sklearn.model_selection import KFold, StratifiedKFold  # TimeSeriesSplit

# from aeon.forecasting.base import ForecastingHorizon  # removido en aeon 1.0.0
# from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import ExpandingWindowSplitter


def instance_partition_index(
    x_data,
    y_data,
    n_splits=5,
    shuffle=True,
    random_state=10,
    stratify=False,
    corr_instances=False,
    window_length_pct=0.5,
    step_length_pct=0.2,
    fh_pct=0.2,
    kwargs=None,
):
    """
    Creates train/test split indices for cross-validation with support for both standard
    and time series-aware splitting strategies.

    Parameters
    ----------
    x_data : array-like
        Feature data with shape (n_samples, ...). Used to determine dataset size.

    y_data : array-like or None
        Target values with shape (n_samples,). Required for stratified sampling.
        Must have the same length as x_data's first dimension.

    n_splits : int, default=5
        Number of folds for standard cross-validation.Ignored when corr_instances=True.

    shuffle : bool, default=True
        Whether to shuffle data before splitting.Ignored when corr_instances=True.

    random_state : int, default=10
        Random seed for reproducible splits when shuffle=True. Ignored when corr_instances=True.

    stratify : bool, default=False
        If True, uses StratifiedKFold to preserve class distribution in each fold.
        Ignored when corr_instances=True.

    corr_instances : bool, default=False
        If True, uses time series-aware splitting (ExpandingWindowSplitter) that
        respects ordered dependencies (in the first dimension) between observations.

    window_length_pct : float, default=0.5
        Initial training window length as a percentage of total data size.
        Only used when corr_instances=True.
        Must be between 0 and 1.

    step_length_pct : float, default=0.2
        Step size between consecutive training windows as a percentage of total data size.
        Only used when corr_instances=True.
        Must be between 0 and 1.

    fh_pct : float, default=0.2
        Forecast horizon length as a percentage of total data size.
        Only used when corr_instances=True.
        Must be between 0 and 1.

    kwargs : dict, default=None
        Additional keyword arguments (not currently used).

    Returns
    -------
    dict
        A nested dictionary with fold indices structured as:
        {
            fold_idx: {
                0: array of train indices,
                1: array of test indices
            },
            ...
        }

    Notes
    -----
    - When corr_instances=True, an expanding window approach is used for time series data
    - The function ensures parameters are within valid ranges
    - For extremely small datasets, it may use LOO cross-validation if n_splits equals size
    - The function can handle both standard and stratified sampling
    - A final fold using all available training data may be added when using time series splitting

    """
    # Validate percentages
    if not (0 < window_length_pct <= 1):
        raise ValueError("window_length_pct must be between 0 and 1.")
    if not (0 < fh_pct <= 1):
        raise ValueError("fh_pct must be between 0 and 1.")
    if not (0 < step_length_pct <= 1):
        raise ValueError("step_length_pct must be between 0 and 1.")

    size = x_data.shape[0]
    # Check if y_data is provided and has the same length as x_data
    if y_data is not None and len(y_data) != size:
        raise ValueError("y_data must have the same length as x_data.")
    x = np.arange(size)
    folds = {}

    if corr_instances:  # we use expanding window
        print(
            "n_splits, shuffle, stratify and random_state are ignored when corr_instances is True."
        )

        # Convert percentage to actual window length (minimum 1)
        window_length = max(1, int(window_length_pct * size))

        # Calculate forecast horizon steps based on percentage
        fh_steps = max(1, int(fh_pct * size))
        # Calculate step length based on percentage
        step_length = max(1, int(step_length_pct * size))

        # Convert to list of consecutive integers starting from 1
        fh_list = list(range(1, fh_steps + 1))

        # Create ForecastingHorizon object
        # fh = ForecastingHorizon(fh_list)
        cv = ExpandingWindowSplitter(
            initial_window=window_length,
            fh=fh_list,  # type: ignore
            step_length=step_length,  # type: ignore
        )
        # Get all CV splits
        splits = list(cv.split(x))

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            # Store each fold in the dictionary using fold index as key
            folds[fold_idx] = {
                0: train_idx,  # 0 for train indices
                1: test_idx,  # 1 for test indices
            }
        # Add final fold if needed (the one that uses all data except the last few points)
        last_test_end = max(splits[-1][1]) + 1 if splits else 0
        if last_test_end < size:
            # Calculate how much data remains
            final_train_idx = np.arange(0, size - fh_steps)
            final_test_idx = np.arange(size - fh_steps, size)
            final_fold_idx = len(splits)
            folds[final_fold_idx] = {
                0: final_train_idx,
                1: final_test_idx,
            }
    else:  # we use standard KFold stratified or not
        if stratify:
            if shuffle:
                skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            else:
                skf = StratifiedKFold(n_splits=n_splits, shuffle=False)

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(x, y_data)):
                folds[fold_idx] = {
                    0: train_idx,  # 0 for train indices
                    1: test_idx,  # 1 for test indices
                }
        else:
            if size > n_splits:
                if shuffle:
                    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
                else:
                    kf = KFold(n_splits=n_splits, shuffle=False)
            elif size == n_splits:
                # This is equivalent to Leave One Out (LOO) cross-validation
                print("Using Leave One Out (LOO) cross-validation. Setting shuffle to False.")
                kf = KFold(n_splits=len(x), shuffle=False)

            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(x)):
                folds[fold_idx] = {
                    0: train_idx,  # 0 for train indices
                    1: test_idx,  # 1 for test indices
                }
    folds_metadata = {
        # "dataset": "SelfRegulationSCP1",
        "creation_date": str(np.datetime64("now")),
        "parameters": {
            "n_splits": n_splits,
            "random_state": random_state,
            "stratify": stratify,
            "corr_instances": corr_instances,
            "window_length_pct": window_length_pct,
            "step_length_pct": step_length_pct,
            "fh_pct": fh_pct,
        },
    }
    return folds, folds_metadata


def serialize_folds(folds, folds_metadata, dataset_name):
    """
    Serializes the folds dictionary to a JSON-compatible format.

    Parameters
    ----------
    folds : dict
        Dictionary containing fold indices.

    Returns
    -------
    str
        Path to the saved JSON file.
    """
    serializable_folds = {}
    for fold_idx, fold_data in folds.items():
        serializable_folds[str(fold_idx)] = {  # Convert keys to strings for JSON
            "train_indices": fold_data[0].tolist(),  # Convert train indices to list
            "test_indices": fold_data[1].tolist(),  # Convert test indices to list
        }
    # Create the complete data structure with both metadata and fold data
    complete_data = {
        "dataset": dataset_name,  # Use the provided dataset name
        "creation_date": folds_metadata.get("creation_date", str(np.datetime64("now"))),
        "parameters": folds_metadata.get("parameters", {}),
        "folds": serializable_folds,  # Include the serialized folds
    }

    # Ensure directory exists
    os.makedirs("results/folds", exist_ok=True)

    # Create filename based on dataset name
    filename = f"results/folds/{dataset_name.lower()}_folds.json"

    # Write the combined data to the JSON file
    with open(filename, "w") as f:
        json.dump(complete_data, f, indent=2)

    print(f"Fold data saved to {filename}")
    return filename


def deserialize_folds(file_path):
    """
    Loads fold information from a JSON file and returns reconstructed fold indices.

    Parameters
    ----------
    file_path : str
        Path to the JSON file containing serialized fold data.

    Returns
    -------
    tuple
        A tuple containing:
        - folds : dict
            Nested dictionary with fold indices structured as:
            {
                fold_idx: {
                    0: array of train indices,
                    1: array of test indices
                },
                ...
            }
        - metadata : dict
            Dictionary with metadata about the folds, including:
            - dataset: name of the dataset
            - creation_date: when the folds were created
            - parameters: parameters used to create the folds
    """
    try:
        # Load the JSON file
        with open(file_path) as f:
            data = json.load(f)

        # Extract metadata
        metadata = {
            "dataset": data.get("dataset", "unknown"),
            "creation_date": data.get("creation_date", ""),
            "parameters": data.get("parameters", {}),
        }

        # Reconstruct folds with proper types
        folds = {}
        for fold_idx, fold_content in data.get("folds", {}).items():
            # Convert string keys back to integers
            fold_idx_int = int(fold_idx)

            # Convert lists back to numpy arrays
            folds[fold_idx_int] = {
                0: np.array(fold_content.get("train_indices", [])),
                1: np.array(fold_content.get("test_indices", [])),
            }

        print(f"Successfully loaded {len(folds)} folds from {file_path}")
        return folds, metadata

    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return {}, {}
    except json.JSONDecodeError:
        print(f"Error: File {file_path} is not valid JSON.")
        return {}, {}
    except Exception as e:
        print(f"Error loading folds from {file_path}: {str(e)}")
        return {}, {}


if __name__ == "__main__":
    # Dummy data for testing
    seed = 42
    # Create feature data: 20 samples with 2 features each
    rng = np.random.default_rng(seed)
    x_data = rng.random((21, 5, 100))  # Random values between 0 and 1

    # Create balanced labels: 10 samples of class 0, 10 samples of class 1
    y_data = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    # Create imbalanced labels: 15 samples of class 0, 5 samples of class 1    # Get all folds
    folds, metadata = instance_partition_index(
        x_data,
        y_data,
        corr_instances=True,
        n_splits=10,
        shuffle=True,
        random_state=seed,
        stratify=True,
        window_length_pct=0.6,
        step_length_pct=0.1,
        fh_pct=0.2,
    )
    # save folds to data/folds/dataset_name
    serialize_folds(folds, metadata, "borrar_test")
    # load folds from data/folds/dataset_name
    folds_loaded, metadata_loaded = deserialize_folds("results/folds/borrar_test_folds.json")

    # compare if the loaded folds are the same as the original ones
    def are_folds_equal(folds1, folds2):
        """Compare two fold dictionaries properly"""
        if set(folds1.keys()) != set(folds2.keys()):
            print("Different fold keys")
            return False

        for fold_idx in folds1:
            if (
                0 not in folds1[fold_idx]
                or 1 not in folds1[fold_idx]
                or 0 not in folds2[fold_idx]
                or 1 not in folds2[fold_idx]
            ):
                print(f"Missing train or test indices in fold {fold_idx}")
                return False

            # Compare training indices
            if not np.array_equal(folds1[fold_idx][0], folds2[fold_idx][0]):
                print(f"Training indices different for fold {fold_idx}")
                return False

            # Compare testing indices
            if not np.array_equal(folds1[fold_idx][1], folds2[fold_idx][1]):
                print(f"Testing indices different for fold {fold_idx}")
                return False

        return True

    assert are_folds_equal(folds_loaded, folds)
    if False:
        for i, fold_data in folds_loaded.items():
            print(f"Fold {i + 1}:")
            print(f"Train indices: {fold_data[0]}")
            print(f"Test indices: {fold_data[1]}")
    if True:
        for fold_idx, fold_data in folds.items():
            print(f"Fold {fold_idx}:")
            # There are two ways to access the data
            print(f"Train indices: {folds[fold_idx][0]}")
            train_indices = fold_data[0]

            # There are two ways to access the data
            print(f"Test indices: {folds[fold_idx][1]}")
            test_indices = fold_data[1]

            X_train, X_test = x_data[train_indices], x_data[test_indices]
            y_train, y_test = y_data[train_indices], y_data[test_indices]
            print(f"X_train shape: {X_train.shape}")
            print(f"X_test shape: {X_test.shape}")
            print(f"y_train shape: {y_train.shape}")
            print(f"y_test shape: {y_test.shape}")
            print()

    # for fold in folds:
    #    print(f"Fold {fold}:")
    #    print(f"Train indices: {folds[fold][0]}")
    #    print(f"Test indices: {folds[fold][1]}")
    #    print()
