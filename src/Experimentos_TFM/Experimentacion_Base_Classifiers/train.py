import os


import time
from datetime import datetime

import numpy as np
import sklearn
from aeon.classification.convolution_based import (
    HydraClassifier,  # noqa: F401
    MiniRocketClassifier,  # noqa: F401
)
from aeon.classification.deep_learning import (  # noqa: F401
    InceptionTimeClassifier,
    IndividualInceptionClassifier,
)
from aeon.classification.dictionary_based import MUSE  # noqa: F401
from aeon.classification.interval_based import TimeSeriesForestClassifier, CanonicalIntervalForestClassifier  # noqa: F401

# For slice timestamps
from ml_edm.utils import check_timestamps
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier 

from components.data.read_dataset import read_datasets
from utils.partitioning import instance_partition_index, serialize_folds

# pip install git+https://github.com/Faiber09/ML_EDM_Multi.git@Desarrollo

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



def get_classifier(classifier_name, random_seed=42, hyperparameters=None):  # noqa: C901
    """
    Return a classifier instance with custom hyperparameters.

    Parameters
    ----------
    classifier_name : str
        Name of the classifier
    random_seed : int, default=42
        Random seed for reproducible results
    hyperparameters : dict, default=None
        Classifier-specific hyperparameters

    Returns
    -------
    sklearn-compatible classifier
        An instance of the requested classifier
    """
    n_jobs = 10 # cambiar si quiero usar mas nucleos por cada clasificador
    # Default hyperparameters for each classifier
    default_params = {
        "MiniRocketClassifier": {
            "random_state": random_seed,
            "n_jobs": n_jobs,
            "n_kernels": 10000,
            "alphas": np.logspace(-3, 4.5, 15),
            #"cv":3
        },
        "HydraClassifier": {
            "random_state": random_seed,
            "n_jobs": n_jobs,
            "alphas": np.logspace(-3, 4.5, 15),
            #"cv":3
        },
        "InceptionTimeClassifier": {
            "random_state": random_seed,
            "batch_size": 64,  # 64 es default
            "n_epochs": 1500,  # 1500 es default
            "n_classifiers": 5,  # clasificadores del ensamble, 5 es default
            "verbose": False,
            "validation_split": 0.2,
        },
        "MUSE": {
            "random_state": random_seed,
            "n_jobs": n_jobs,
            "alphas": np.logspace(-3, 4.5, 15),
            #"cv":3
        },
        "TimeSeriesForestClassifier": {
            "random_state": random_seed,
            "n_jobs": n_jobs,
            "n_estimators": 50,
            "base_estimator": DecisionTreeClassifier(random_state=random_seed),
            "tune": True,  # Activar el tuning si el clasificador esta envuelto en GridSearchCV
            "tune_params": {  # Parámetros a tunear (aunque hay algunso por defecto)
                #"n_estimators": [10, 50, 100, 200], # para tunear algun parametro del clasificador, por ejemplo
                "base_estimator__max_depth": [3, 6, 9, 12, 15],
            },
            "cv": 3,  # Folds para el tuning
        },
        "CanonicalIntervalForestClassifier": {
            "random_state": random_seed,
            "n_jobs": n_jobs,
            "tune": False,  # Activar el tuning si el clasificador esta envuelto en GridSearchCV
            #"base_estimator": DecisionTreeClassifier(random_state=random_seed, criterion="entropy"),
            "n_estimators":200,
            "tune_params": {  
                #"n_estimators": [10, 50, 100, 200], # para tunear algun parametro del clasificador, por ejemplo
                "base_estimator__max_depth": [5, 10, 15, None],
            },
            "cv": 3,  # Folds para el tuning
        },
    }

    # Use default parameters if none provided
    if hyperparameters is None:
        hyperparameters = {}

    # Get default parameters for this classifier type
    clf_defaults = default_params.get(classifier_name, {})

    # Merge defaults with provided hyperparameters (provided take precedence)
    params = {**clf_defaults, **hyperparameters}

    # Create classifier with parameters
    if classifier_name == "MiniRocketClassifier":
        # Create a copy of the params to modify
        minirocket_params = params.copy()
        # Extract parameters meant for RidgeClassifierCV
        if "alphas" in minirocket_params:
            alphas = minirocket_params.pop("alphas")
            cv = minirocket_params.pop("cv") if "cv" in minirocket_params else None
            # Create custom ridge estimator
            custom_ridge = RidgeClassifierCV(alphas=alphas, cv=cv)
            minirocket_params["estimator"] = custom_ridge
        # Pass only the valid parameters to MiniRocketClassifier
        return MiniRocketClassifier(**minirocket_params)

    elif classifier_name == "HydraClassifier":
        # HydraClassifier requires a specific set of parameters
        # Create a copy of the params to modify
        hydra_params = params.copy()
        # Extract parameters meant for RidgeClassifierCV
        if "alphas" in hydra_params:
            alphas = hydra_params.pop("alphas")
            cv = hydra_params.pop("cv") if "cv" in hydra_params else None
            # Create custom ridge estimator
            custom_ridge = RidgeClassifierCV(alphas=alphas, cv=cv)
            hydra_params["estimator"] = custom_ridge
        return HydraClassifier(**hydra_params)

    elif classifier_name == "InceptionTimeClassifier":
        return InceptionTimeClassifier(**params)

    elif classifier_name == "MUSE":
        # Create a copy of the params to modify
        muse_params = params.copy()
        # Extract parameters meant for RidgeClassifierCV
        if "alphas" in muse_params:
            alphas = muse_params.pop("alphas")
            cv = muse_params.pop("cv") if "cv" in muse_params else None
            # Create custom ridge estimator
            custom_ridge = RidgeClassifierCV(alphas=alphas, cv=cv)
            muse_params["estimator"] = custom_ridge
            # Ensure we can get probabilities if needed
            # if muse_params.get("support_probabilities", False):
            #    print(
            #        "Warning: Using RidgeClassifierCV with support_probabilities=True may not work as expected."  # noqa: E501
            #    )
        return MUSE(**muse_params)

    elif classifier_name == "TimeSeriesForestClassifier":
        # Create a clean copy of parameters
        clean_params = params.copy()
        
        # Remove parameters that are not valid for TimeSeriesForestClassifier
        tuning_keys = ["tune", "tune_params", "cv"]
        for key in tuning_keys:
            if key in clean_params:
                clean_params.pop(key)
        
        # Explicitly initialize a base_estimator
        clean_params["base_estimator"] = DecisionTreeClassifier(
            random_state=clean_params.get("random_state", 42)
        )
        
        # Check if tuning is required
        if params.get("tune", False):
            # Extract tuning parameters
            tune_params = params.get("tune_params", {})
            cv = params.get("cv", 3)

            # For tuning base_estimator parameters, create separate grid points
            if "base_estimator__max_depth" in tune_params:
                max_depths = tune_params.pop("base_estimator__max_depth")
                param_grid = []
                
                for depth in max_depths:
                    # Create a decision tree with specific depth
                    tree = DecisionTreeClassifier(
                        max_depth=depth,
                        random_state=clean_params.get("random_state", 42)
                    )
                    
                    # Create a grid point with this specific tree
                    param_grid.append({
                        "base_estimator": [tree]
                    })
            else:
                # Default parameters for tuning
                param_grid = {"n_estimators": [50, 100, 200]}

            # Create base classifier with clean parameters
            base_clf = TimeSeriesForestClassifier(**clean_params)

            # Create and return GridSearchCV
            return GridSearchCV(
                estimator=base_clf,
                param_grid=param_grid,
                cv=cv,
                n_jobs=clean_params.get("n_jobs", 1),
                verbose=1,
            )
        else:
            # Return classifier without tuning
            return TimeSeriesForestClassifier(**clean_params)
        

    elif classifier_name == "CanonicalIntervalForestClassifier":
        clean_params = params.copy()
        tuning_keys = ["tune", "tune_params", "cv"]
        for key in tuning_keys:
            if key in clean_params:
                clean_params.pop(key)

        if params.get("tune", False):

            tune_params_grid = params.get("tune_params", {})
            cv_folds = params.get("cv", 3)

            if not tune_params_grid:
                tune_params_grid = {
                    "base_estimator__max_depth": [5, 10, 20, None], # Ejemplo para max_depth
                    #"min_interval_length": [3, 5, 7], # Ejemplo de parámetro de TSF
                }
            
            # Asegurarse que base_estimator en clean_params es una instancia
            if not isinstance(clean_params.get("base_estimator"), DecisionTreeClassifier):
                clean_params["base_estimator"] = DecisionTreeClassifier(random_state=clean_params.get("random_state"))


            base_clf = CanonicalIntervalForestClassifier(**clean_params)

            return GridSearchCV(
                estimator=base_clf,
                param_grid=tune_params_grid,
                cv=cv_folds,
                n_jobs=clean_params.get("n_jobs", 1), # n_jobs para GridSearchCV
                verbose=1,
            )
        else:

            return CanonicalIntervalForestClassifier(**clean_params)
            
    else:
        raise ValueError(f"Classifier {classifier_name} not supported.")




def generate_timestamps(max_t, sampling_ratio=0.05, min_value=9):
    """
    Generate timestamps for slicing time series data.

    Parameters
    ----------
    max_t : int
        Length of the time series
    sampling_ratio : float, default=0.05
        Sampling ratio for timestamps
    min_value : int, default=9
        Minimum value for any timestamp (this is important for some classifiers)
    """
    timestamps = [int(max_t * (sampling_ratio * i)) for i in range(1, int(1 / sampling_ratio) + 1)]

    # Add the final timestamp if not already included
    if max_t not in timestamps:
        timestamps.append(max_t)

    # Filter out timestamps less than min_value
    timestamps = [ts for ts in timestamps if ts >= min_value]

    if not timestamps:
        timestamps = [min_value]

    # Remove duplicates and zeros
    timestamps = check_timestamps(timestamps)

    return timestamps


def train_and_evaluate(x_data, y_data, folds, timestamps, classifier_name, base_seed, hyperparameters):
    """Train and evaluate for each fold and timestamp."""
    train_accuracies = []
    test_accuracies = []
    train_predictions = []
    test_predictions = []
    train_probabilities = []
    test_probabilities = []
    train_true_labels = []
    test_true_labels = []
    best_alphas = []  # to store the best alphas for each fold, ts (minirocket and hydra)
    best_epochs = []  # to store the best epochs for each fold, ts (inception)
    best_grid_params = []  # to store the best grid params for each fold, ts (TimeSeriesForest)

    for i, fold_data in folds.items():
        initial_time_res = time.perf_counter()

        # get the train and test indices
        train_indices = fold_data[0]
        test_indices = fold_data[1]

        # get the train and test data
        x_train, y_train = x_data[train_indices], y_data[train_indices]
        x_test, y_test = x_data[test_indices], y_data[test_indices]

        # save the true labels for each fold
        train_true_labels.append(y_train)
        test_true_labels.append(y_test)

        fold_train_accuracies = []
        fold_test_accuracies = []
        fold_train_predictions = []
        fold_test_predictions = []
        fold_train_probabilities = []
        fold_test_probabilities = []
        fold_best_alphas = []
        fold_best_epochs = []
        fold_best_grid_params = []

        # instanciamos un clasificador con distinta seed para cada resampling, así cada resampling tien una inicializacion diferente
        # si quisieramos exactamente la misma inicializacion eliminamos el "i"
        semilla = base_seed + i
        fold_clf = get_classifier(
            classifier_name,
            random_seed=semilla,
            hyperparameters=hyperparameters,
        )
        print(f"Resampling {i}: inicializando clasificador con random_state={semilla}")
        print("Training for each timestamp:")
        max_t = x_data.shape[-1]
        for ts_idx, ts in enumerate(timestamps):  # noqa: B007
            print(f"  Training model for timestamp {ts}/{max_t} ({ts / max_t:.2%})")
            x_train_ts = x_train[:, :, :ts]
            x_test_ts = x_test[:, :, :ts]

            # Create and fit a new model for this timestamp
            ts_clf = sklearn.base.clone(fold_clf)

            # Fit the model with sliced data
            ts_clf.fit(x_train_ts, y_train)

            # ==== Regularization Parameters Extraction ====
            best_alpha, best_epoch, grid_params = extract_hyperparameters(ts_clf)

            fold_best_epochs.append(best_epoch)
            fold_best_alphas.append(best_alpha)
            fold_best_grid_params.append(grid_params)

            # === END of Regularization Parameters Extraction ====

            # Make predictions
            train_prediction_ts = ts_clf.predict(x_train_ts)
            test_prediction_ts = ts_clf.predict(x_test_ts)

            # Store predictions
            fold_train_predictions.append(train_prediction_ts)
            fold_test_predictions.append(test_prediction_ts)

            # Get probabilities if available
            try:
                train_proba_ts = ts_clf.predict_proba(x_train_ts)
                test_proba_ts = ts_clf.predict_proba(x_test_ts)
                fold_train_probabilities.append(train_proba_ts)
                fold_test_probabilities.append(test_proba_ts)
            except (ValueError, AttributeError, NotImplementedError) as e:
                print(f"  Warning: Classifier doesn't support predict_proba: {e}")
                fold_train_probabilities.append(None)
                fold_test_probabilities.append(None)

            # Calculate accuracies
            train_ts_accuracy = ((train_prediction_ts == y_train).astype(int)).mean()
            fold_train_accuracies.append(train_ts_accuracy)

            test_ts_accuracy = ((test_prediction_ts == y_test).astype(int)).mean()
            fold_test_accuracies.append(test_ts_accuracy)
            print(f"    Test accuracy: {test_ts_accuracy:.4f}")

        # Store results for this fold
        train_predictions.append(fold_train_predictions)
        test_predictions.append(fold_test_predictions)
        train_probabilities.append(fold_train_probabilities)
        test_probabilities.append(fold_test_probabilities)
        train_accuracies.append(fold_train_accuracies)
        test_accuracies.append(fold_test_accuracies)
        best_alphas.append(fold_best_alphas)
        best_epochs.append(fold_best_epochs)
        best_grid_params.append(fold_best_grid_params)

        end_time_res = time.perf_counter()
        elapsed_time_res = end_time_res - initial_time_res
        print(f"Training for resampling completed in {elapsed_time_res:.2f} seconds")
        

        # CAUTION: Remove break to train on all folds
        # break  # just for quick experimentation 

    results = {
        "train_accuracies": np.array(train_accuracies),
        "test_accuracies": np.array(test_accuracies),
        "train_predictions": np.array(train_predictions, dtype=object),
        "test_predictions": np.array(test_predictions, dtype=object),
        "train_probabilities": np.array(train_probabilities, dtype=object),
        "test_probabilities": np.array(test_probabilities, dtype=object),
        "train_true_labels": np.array(train_true_labels, dtype=object),
        "test_true_labels": np.array(test_true_labels, dtype=object),
        "timestamps": np.array(timestamps),
        "best_alphas": np.array(best_alphas, dtype=object),
        "best_epochs": np.array(best_epochs, dtype=object),
        "best_grid_params": np.array(best_grid_params, dtype=object),
    }

    return results


def extract_hyperparameters(ts_clf):  # noqa: C901
    """
    Extract hyperparameters from all supported classifier types.

    Parameters
    ----------
    ts_clf : object
        Trained classifier instance

    Returns
    -------
    tuple
        (best_alpha, best_epoch, grid_params)
    """
    best_alpha = None
    best_epoch = None
    grid_params = None

    # Case 1: MiniRocketClassifier
    if isinstance(ts_clf, MiniRocketClassifier):
        try:
            ridge_clf = ts_clf.pipeline_.steps[-1][1]
            if hasattr(ridge_clf, "alpha_"):
                best_alpha = ridge_clf.alpha_
                print(f"    Best alpha: {best_alpha}")
        except (AttributeError, IndexError) as e:
            print(f"    Error extracting alpha: {e}")

    # Case 2: HydraClassifier
    elif isinstance(ts_clf, HydraClassifier):
        try:
            ridge_clf = ts_clf._clf.steps[-1][1]
            if hasattr(ridge_clf, "alpha_"):
                best_alpha = ridge_clf.alpha_
                print(f"    Best alpha: {best_alpha}")
        except (AttributeError, IndexError) as e:
            print(f"    Error extracting alpha: {e}")

    # Case 3: MUSE
    elif isinstance(ts_clf, MUSE):
        try:
            ridge_clf = ts_clf.clf
            if isinstance(ridge_clf, RidgeClassifierCV) and hasattr(ridge_clf, "alpha_"):
                best_alpha = ridge_clf.alpha_
                print(f"    Best alpha: {best_alpha}")
        except (AttributeError, IndexError) as e:
            print(f"    Error extracting alpha from MUSE: {e}")

    # Case 4: InceptionTimeClassifier
    elif isinstance(ts_clf, InceptionTimeClassifier):
        try:
            for classifier in ts_clf.classifiers_:
                if hasattr(classifier, "best_epoch_"):
                    best_epoch = classifier.best_epoch_
                    print(f"    Best epoch: {best_epoch}")
                    break
        except (AttributeError, IndexError) as e:
            print(f"    Error extracting best epoch: {e}")

    # Case 5: IndividualInceptionClassifier
    elif isinstance(ts_clf, IndividualInceptionClassifier):
        try:
            if hasattr(ts_clf, "best_epoch_"):
                best_epoch = ts_clf.best_epoch_
                print(f"    Best epoch: {best_epoch}")
        except AttributeError as e:
            print(f"    Error extracting best epoch: {e}")

    # Case 6: GridSearchCV (for any classifier wrapped in GridSearchCV)
    if isinstance(ts_clf, GridSearchCV):
        try:
            grid_params = ts_clf.best_params_
            print(f"    Mejores parámetros: {grid_params}")
        except (AttributeError, IndexError) as e:
            print(f"    Error extrayendo mejores parámetros: {e}")

    return best_alpha, best_epoch, grid_params


def save_results(
    results, dataset_name, classifier_name, folds_metadata=None, runtime=None, hyperparameters=None
):
    """Save results to a file."""
    # Create directory if it doesn't exist
    save_dir = os.path.join("results", "base_results_final")
    os.makedirs(save_dir, exist_ok=True)

    # Generate timestamp for unique filenames
    marca_tiempo = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Add metadata to results
    complete_results = results.copy()
    complete_results.update(
        {
            "dataset": dataset_name,
            "classifier": classifier_name,
            "folds_metadata": folds_metadata,
            "hyperparameters": hyperparameters,
            "runtime": runtime,
        }
    )

    # Save as numpy file
    save_path = os.path.join(save_dir, f"{dataset_name}_{classifier_name}_{marca_tiempo}.npz")
    np.savez(save_path, **complete_results)
    print(f"Results saved to {save_path}")
    print(f"Hyperparameters: {hyperparameters}")

    return save_path


def train(
    dataset_name="SelfRegulationSCP1",
    classifier_name="MiniRocketClassifier",
    sampling_ratio=0.05,
    hyperparameters=None,
    n_splits=3, # number of resamplings
    random_seed=42,
    stratify=True,
    test_size = 0.2,
    corr_instances=False,
    window_length_pct=0.5,
    step_length_pct=0.1,
    fh_pct=0.1,
):
    """
    Train time series classifiers for a given dataset.

    Parameters
    ----------
    dataset_name : str, default="SelfRegulationSCP1"
        Name of the dataset to use
    classifier_name : str, default="MiniRocketClassifier"
        Name of the classifier to use
    random_seed : int, default=42
        Random seed for reproducibility
    n_splits : int, default=3
        Number of cross-validation splits
    sampling_ratio : float, default=0.05
        Sampling ratio for timestamps

    """
    print(f"Training {classifier_name} on {dataset_name}")

    # 1. Load and prepare data
    x_data, y_data, meta = read_datasets(dataset_name)
    print(f"Data shape: {x_data.shape}, {y_data.shape}")

    # 2. Get folds for cross-validation
    folds, folds_metadata = instance_partition_index(
        x_data,
        y_data,
        n_splits=n_splits,
        random_state=random_seed,
        stratify=stratify,
        test_size = test_size,
        corr_instances=corr_instances,
        window_length_pct=window_length_pct,
        step_length_pct=step_length_pct,
        fh_pct=fh_pct,
    )

    # 3. Save resamplings indexes for reproducibility (optional)
    #serialize_folds(folds, folds_metadata, dataset_name)

    # 4. Get timestamps for slicing
    max_t = x_data.shape[-1]
    timestamps = generate_timestamps(max_t, sampling_ratio)

    # 6. Train and evaluate
    initial_time = time.perf_counter()
    results = train_and_evaluate(x_data, y_data, folds, timestamps, classifier_name, random_seed, hyperparameters)
    end_time = time.perf_counter()
    elapsed_time = end_time - initial_time
    print(f"Training completed in {elapsed_time:.2f} seconds")

    # 7. Save results
    # Store hyperparameters in metadata
    if hyperparameters is None:
        hyperparameters = {}

    save_results(
        results, dataset_name, classifier_name, folds_metadata, elapsed_time, hyperparameters
    )

    return results


if __name__ == "__main__":

    lista_datasets_pequeños = [
        "Libras",  # 45 timepoints
        "AtrialFibrillation",
        "BasicMotions",
        "RacketSports",  # 30 timepoints
        "ERing",
        "Epilepsy",  
        # "PenDigits",      # Very Small only 8 timepoints ELIMINADO POR COMPLETO DE EXPERIMENTACION
        "StandWalkJump",  # solo 30 instancias
        "UWaveGestureLibrary",
        "NATOPS",  # 51 timepoints
        "Handwriting",
        "FingerMovements",  # 50 timepoints
        "ArticularyWordRecognition",
        "HandMovementDirection",
        "Cricket",
        "SelfRegulationSCP1",
        #"LSST",  # 36 timepoints  Large
        # "EthanolConcentration", # Large
        #"SelfRegulationSCP2",  # large NO SIRVE PARA TRIGGERS
        # "Heartbeat",  # Large
        # "PhonemeSpectra", # Large
        # "EigenWorms",        # Large
        # "DuckDuckGeese",    # Large  NO SIRVE PARA TRIGGERS
        # "PEMS-SF", # Large
        # "MotorImagery", # Large  NO SIRVE PARA TRIGGERS
        # "FaceDetection",   # Large 
    ]
    lista_datasets_grandes = [
        "LSST",
        "EthanolConcentration",
        "Heartbeat",  # Large
        "PhonemeSpectra", # Large
        "EigenWorms",        # Large
        "PEMS-SF", # Large
        "FaceDetection",
        "SelfRegulationSCP2",    # NO SIRVE PARA TRIGGERS                       
        "DuckDuckGeese",    # Large # NO SIRVE PARA TRIGGERS   
        "MotorImagery", # Large # NO SIRVE PARA TRIGGERS   
        ] 

    # lista de clasificadores para experimentos
    classifiers = [
        #"MiniRocketClassifier",
        #"HydraClassifier",
        #"CanonicalIntervalForestClassifier",
        #"InceptionTimeClassifier",
        "MUSE",
        #"TimeSeriesForestClassifier",
    ]
    #lista_datasets = lista_datasets_pequeños + lista_datasets_grandes

    # lista de datasets para experimentos
    lista_datasets = [
        "FaceDetection"
        ]

    import logging
    import traceback
    from datetime import datetime

    # Configurar logging
    logging.basicConfig(
        filename=f"experiments_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    total_combinations = len(classifiers) * len(lista_datasets)
    successful = 0
    failed = 0

    # Para cada combinación
    for classifier in classifiers:
        for dataset in lista_datasets:
            try:
                logging.info(f"Iniciando experimento: {classifier} en {dataset}")
                print(f"\n{'=' * 50}\nIniciando: {classifier} en {dataset}\n{'=' * 50}")
                # En caso de que quiera meter algunos hyperparametros distintos para algun dataset
                # hyperparameters = {"alphas":np.logspace(-3, 5, 15)} if dataset =="AtrialFibrillation" else {}
                # Ejecutar el experimento
                seed = 42
                train(
                    dataset_name=dataset,
                    classifier_name=classifier,
                    sampling_ratio=0.05, # for slicing the total TS
                    hyperparameters=None, # si meto None se usan hyperparametros por defecto en get_classifier
                    n_splits=10,  # number of resamplings
                    random_seed=seed,
                    stratify=True,
                    test_size = 0.2,
                    corr_instances=False, # if set false the remaining parameters are irrelevant
                    window_length_pct=0.5,
                    step_length_pct=0.1,
                    fh_pct=0.1,
                )

                logging.info(f"Experimento completado: {classifier} en {dataset}")
                successful += 1

            except Exception as e:
                error_msg = f"ERROR en {classifier} - {dataset}: {str(e)}"
                stack_trace = traceback.format_exc()

                print(f"\n❌ {error_msg}")
                logging.error(error_msg)
                logging.error(f"Stack trace: {stack_trace}")

                failed += 1

            finally:
                # Mostrar progreso
                completed = successful + failed
                print(
                    f"\nProgreso: {completed}/{total_combinations} ({completed / total_combinations:.1%})"
                )
                print(f"Exitosos: {successful}, Fallidos: {failed}")

    print(f"\n{'=' * 50}")
    print(f"EXPERIMENTOS COMPLETADOS: {successful}/{total_combinations}")
    print(f"EXPERIMENTOS FALLIDOS: {failed}/{total_combinations}")
    print(f"{'=' * 50}")
    logging.info(
        f"RESUMEN FINAL - Exitosos: {successful}/{total_combinations}, Fallidos: {failed}/{total_combinations}"
    )
