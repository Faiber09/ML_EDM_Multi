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
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    ShuffleSplit,
) # TimeSeriesSplit

# from aeon.forecasting.base import ForecastingHorizon  # removido en aeon 1.0.0
# from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import ExpandingWindowSplitter


import json
import os
import numpy as np

# Para series temporales
from sktime.forecasting.model_selection import ExpandingWindowSplitter
# Para particionados no temporales
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    ShuffleSplit,
)

def instance_partition_index(
    x_data,
    y_data,
    n_splits=10,
    random_state=42,
    stratify=False,
    test_size = 0.2,
    corr_instances=False,
    window_length_pct=0.5,
    step_length_pct=0.2,
    fh_pct=0.2,
    kwargs=None,
):
    """
    Crea índices de particionado train/test para validación cruzada, con soporte para
    estrategias estándar y para series temporales.

    Parámetros
    ----------
    x_data : array-like
        Datos de características con shape (n_samples, ...). Se usa para determinar el tamaño del dataset.
    
    y_data : array-like o None
        Valores objetivo con shape (n_samples,). Es requerido para el muestreo estratificado y debe
        tener la misma longitud que x_data.
    
    n_splits : int, default=10
        Número de particiones. En el caso de resampleo aleatorio se generarán n_splits particiones.
    
    
    random_state : int, default=42
        Semilla para reproducibilidad cuando shuffle es True. Se ignora cuando corr_instances es True.
    
    stratify : bool, default=False
        Si True, se utiliza particionado estratificado (resampleo 80-20) para preservar la proporción de clases.
    
    test_size : float, default = 0.2
        porción reservada para test cuando se hace el train/test split. Se usa el mismo valor con o sin Stratify
    
    corr_instances : bool, default=False
        Si True, se utiliza particionado consciente de series temporales (ExpandingWindowSplitter),
        respetando el orden de las observaciones.
    
    window_length_pct : float, default=0.5
        Longitud inicial de la ventana de entrenamiento como porcentaje del total (usado en series temporales).
    
    step_length_pct : float, default=0.2
        Tamaño del paso entre ventanas consecutivas como porcentaje del total (usado en series temporales).
    
    fh_pct : float, default=0.2
        Longitud del horizonte de pronóstico como porcentaje del total (usado en series temporales).
    
    kwargs : dict, default=None
        Argumentos adicionales (actualmente no usados).

    Retorna
    -------
    dict
        Un diccionario con los índices de particionado por fold, con la estructura:
        {
            fold_idx: {
                0: array de índices de train,
                1: array de índices de test
            },
            ...
        }
    """
    # Validar porcentajes
    if not (0 < window_length_pct <= 1):
        raise ValueError("window_length_pct must be between 0 and 1.")
    if not (0 < fh_pct <= 1):
        raise ValueError("fh_pct must be between 0 and 1.")
    if not (0 < step_length_pct <= 1):
        raise ValueError("step_length_pct must be between 0 and 1.")

    size = x_data.shape[0]
    # Verificar que y_data tenga la misma longitud que x_data (si se proporciona)
    if y_data is not None and len(y_data) != size:
        raise ValueError("y_data must have the same length as x_data.")
    x = np.arange(size)
    folds = {}

    if corr_instances:
        print(
            "n_splits, shuffle, stratify y random_state se ignoran cuando corr_instances es True."
        )
        window_length = max(1, int(window_length_pct * size))
        fh_steps = max(1, int(fh_pct * size))
        step_length = max(1, int(step_length_pct * size))
        fh_list = list(range(1, fh_steps + 1))

        cv = ExpandingWindowSplitter(
            initial_window=window_length,
            fh=fh_list,  # type: ignore
            step_length=step_length,  # type: ignore
        )
        splits = list(cv.split(x))

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            folds[fold_idx] = {0: train_idx, 1: test_idx}
        # Agregar fold final si quedan datos sin usar
        last_test_end = max(splits[-1][1]) + 1 if splits else 0
        if last_test_end < size:
            final_train_idx = np.arange(0, size - fh_steps)
            final_test_idx = np.arange(size - fh_steps, size)
            final_fold_idx = len(splits)
            folds[final_fold_idx] = {0: final_train_idx, 1: final_test_idx}
    else:
        if stratify:
            # Resampleo estratificado 80-20
            sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
            for fold_idx, (train_idx, test_idx) in enumerate(sss.split(x, y_data)):
                folds[fold_idx] = {0: train_idx, 1: test_idx}
        else:
            # Resampleo aleatorio no estratificado 80-20
            ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
            for fold_idx, (train_idx, test_idx) in enumerate(ss.split(x)):
                folds[fold_idx] = {0: train_idx, 1: test_idx}

    folds_metadata = {
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
        random_state=seed,
        stratify=True,
        test_size = 0.2,
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
