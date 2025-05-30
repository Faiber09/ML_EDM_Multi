import os 
import copy
import numpy as np

from ._base import BaseTimeClassifier

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

from .features_engineering.features_extraction import Feature_extractor
#from ..trigger import *
#from ..utils import *

from warnings import warn

# TODO : Multivariate data
# TODO : Enrich / change feature extraction
# TODO : Add calibration
# TODO : deal with NaN in dataset ? (Although HGBoost is nan compatible)
# TODO : Add set_params, setters etc... Check integrity issues doing so
# TODO : Optimize decision threshold
# TODO : Verbose, loadbar?
# TODO : implement sparse matrix compatibility
# TODO : Make classes sklearn estimators

class ClassifiersCollection(BaseTimeClassifier):
    """
    A class containing a list of classifiers to train on time series of incrementing lengths. Can be used along with a
    trigger model object such as an EconomyGamma instance to do tasks of early classification.

    Parameters:
    -----------

        base_classifier : classifier instance, default=sklearn.ensemble.HistGradientBoostingClassifier()
            Classifier instance to be cloned and trained for each input length.
        timestamps : numpy.ndarray, default=None
            Array containing the numbers of time measurements/input length that each classifier is trained on.
            Argument 'nb_classifiers' is deduced from the length of this list.
        sampling_ratio : float, default = None
            Ignored if 'timestamps' is defined.
            Foat number between 0 and 1, define frequency at which 'timestamps' are spaced.
        min_length : int, default = None
            Define the minimum serie length for the first classifier to operate on.
        feature_extraction : dict, default = None 
            Either a dictionnary containg one of method ['minirocket', 'weasel2.0', 'tsfresh'] and eventually a 'params'
            to define method parametes as a dict and a 'path' key to define where to save features matrices if desired.
        calibration : boolean, default = True
            Whether or not to use post-hoc calibration (Platt scaling) for each classifier.
        classifiers : numpy.ndarray, default=None
            List or array containing the classifier instances to be trained. Argument 'nb_classifiers' is deduced from
            the length of this list.
        classsifiers_requ_2d : boolean, default=False
            Whether the classifier requires 2D input data or not.
        feature_extractor_requ_2d : boolean, default=False
            Whether the feature extractor requires 2D input data or not.
        calibrator_requ_2d : boolean, default=False
            Whether the calibrator requires 2D input data or not for its fit method.

    Attributes:
    -----------

        nb_classifiers: int, default=20
            Number of classifiers to be trained. If the number is inferior to the number of measures in the training
            time series, the models input lengths will be equally spaced from max_length/n_classifiers to max_length.
        class_prior: numpy.ndarray
            Class prior probabilities vector obtained from the training set. The order of the classes is detailed in the
            'classes_' argument
        classes_:
            Array containing the name of each classes ordered the same way as the 'class_prior' argument. Obtained from
            the 'classes_' argument of the first classifier in the 'classifiers' list of the ChronologicalClassifiers
            object.
        max_series_length: int
            Maximum number of measurements contained by a time series in the training set.
    """

    def __init__(self,
                 base_classifier=None,
                 timestamps=None,
                 sampling_ratio=None,
                 min_length=None,
                 feature_extraction=None,
                 calibration=True,
                 calibration_method='sigmoid', # consider 'isotonic' (for + than 1000 samples)
                 classifiers=None,
                 classifiers_requ_2d=False,
                 feature_extractor_requ_2d=False,
                 calibrator_requ_2d=False,
                 random_state=44):  
        
        super().__init__(timestamps, 
                         sampling_ratio,
                         min_length)
        
        self.base_classifier = base_classifier
        self.classifiers = classifiers

        self.feature_extraction = feature_extraction
        self.calibration = calibration
        self.calibration_method = calibration_method
        self.classifiers_requ_2d = classifiers_requ_2d
        self.feature_extractor_requ_2d = feature_extractor_requ_2d
        self.calibrator_requ_2d = calibrator_requ_2d
        self.random_state = random_state

    def __getitem__(self, item):
        return self.classifiers[item]

    def __len__(self):
        return self.nb_classifiers

    def _fit(self, X, y, *args, **kwargs):
        """
        This method fits every classifier in the ChronologicalClassifiers object by truncating the time series of the
        training set to the input lengths contained in the attribute models_input_lengths'. The prior probabilities are
        also saved.

        Parameters:
            X: np.ndarray
                Training set of matrix shape (N, D, T) where:
                    N is the number of time series
                    D is the number of dimensions of each time series
                    T is the commune length of all complete time series
            y: nd.ndarray
                List of the N corresponding labels of the training set.
        """
        # check classifiers list validity
        if self.classifiers is not None:
            if not isinstance(self.classifiers, list):
                raise TypeError("Argument 'classifiers' should be a list of classifier objects.")
            if len(self.classifiers) == 0:
                raise ValueError("List argument 'classifiers' is empty.")
            if self.base_classifier is not None:
                warn("Both base_classifier and classifiers arguments are defined,"
                     " in that case the base_classifier argument will be ignored.")
        else:
            if self.base_classifier is None:
                self.base_classifier = HistGradientBoostingClassifier(random_state=self.random_state)
                warn("Using 'base_classifier = HistGradientBoostingClassifier() by default.")

            self.classifiers = [copy.deepcopy(self.base_classifier) for _ in range(self.nb_classifiers)]

        # feature_extraction check
        if self.feature_extraction:
            if isinstance(self.feature_extraction, dict) and \
                'method' in self.feature_extraction.keys():
                
                if self.feature_extraction['method'] not in ['minirocket', 'weasel2.0', 'tsfresh', 'hydra']:
                    raise ValueError("Argument 'method' from 'feature_extraction' should be one of "
                                    "['minirocket', 'weasel2.0', 'tsfresh', 'hydra']")
            elif not isinstance(self.feature_extraction, str):
                raise ValueError("Argument 'feature_extraction' should be one of dictionnary "
                                "or string (path from which to retreive already computed features)")

        # FEATURE EXTRACTION AND FITTING
        self.extractors = []
        for i, ts_length in enumerate(self.timestamps):
            Xt = X[..., :ts_length]
            if self.feature_extraction:
                # If feature_extractor_requ_2d is True, the input data is reshaped to 2D.            
                if self.feature_extractor_requ_2d:
                    Xt = Xt.reshape(X.shape[0], -1)
                scale = True if self.feature_extraction['method'] == 'minirocket' else False
                extractor = Feature_extractor(self.feature_extraction['method'], scale,
                                              kwargs=self.feature_extraction['params'])
                extractor.fit(Xt, y)
                self.extractors.append(extractor)
                Xt = extractor.transform(Xt)

            # Prepare the input format for the classifier based on the requires_2d flag.
            # Calibration: split the training data if calibration is enabled.
            if self.calibration:
                Xt_clf, X_calib, y_clf, y_calib = train_test_split(
                    Xt, y, test_size=0.3, stratify=y, random_state=self.random_state
                )
            else:
                Xt_clf, y_clf = Xt, y # keep all training samples

            # If classifiers_requ_2d is True, the input data is reshaped to 2D.
            if self.classifiers_requ_2d:
                Xt_clf = Xt_clf.reshape(Xt_clf.shape[0], -1)
            # Fit the classifier for the current timestamp.
            self.classifiers[i].fit(Xt_clf, y_clf, **kwargs)


            # If calibration is enabled, perform calibration on the fitted classifier.
            if self.calibration:
                # if calibrator_requ_2d is True, the input data is reshaped to 2D.
                if self.calibrator_requ_2d:
                    X_calib = X_calib.reshape(X_calib.shape[0], -1)
                calib_clf = CalibratedClassifierCV(self.classifiers[i], cv='prefit', method=self.calibration_method)
                self.classifiers[i] = calib_clf.fit(X_calib, y_calib)

        return self

    def _predict_proba(self, grouped_X, cost_matrices=None):
        """
        Predict a dataset of time series of various lengths using the right classifier in the ChronologicalClassifiers
        object. If a time series has a different number of measurements than the values in 'models_input_lengths', the
        time series is truncated to the closest compatible length. If its length is shorter than the first length in
        'models_input_lengths', the prior probabilities are used. Returns the class probabilities vector of each series.
        Parameters:
            X: np.ndarray
            Dataset of time series of various sizes to predict. An array of size (N*max_T) where N is the number of
            time series, max_T the max number of measurements in a time series and where empty values are filled with
            nan. Can also be a pandas DataFrame or a list of lists.
        Returns:
            np.ndarray containing the classifier class probabilities array for each time series in the dataset.
        """
        # Return prior if no classifier fitted for time series this short, 
        # predict with classifier otherwise
        predictions = []
        returned_priors = False
        for length, series in grouped_X.items():
            if length < self.timestamps[0]:
                predictions.append(np.ones((len(series), len(self.class_prior))) * self.class_prior)
                returned_priors = True
            else:
                # We are assuming that length match one of the timestamps
                
                # CAREFUL: This is the corrected timestamp-to-classifier mapping ADDED
                clf_idx = np.where(self.timestamps == length)[0][0]  # original code
                # IMPORTANT: This is the corrected timestamp-to-classifier mapping
                # if hasattr(self, "timestamp_to_idx") and length in self.timestamp_to_idx:
                #     # Use the mapping created by ECDIRE
                #     clf_idx = self.timestamp_to_idx[length]
                # else:
                #     # Use the original logic - find the closest classifier for this length
                #     clf_idx = np.searchsorted(self.timestamps, length, side='right') - 1

                # Ensure it's a numpy array; expected shape: (N, D, length)
                series = np.array(series)
                if self.feature_extraction:
                # Use precomputed features if the feature_extraction parameter is a directory.
                    if os.path.isdir(str(self.feature_extraction)):
                        fts_idx = clf_idx
                        if hasattr(self, "prev_models_input_lengths"):
                            fts_idx = np.where(self.prev_models_input_lengths == length)[0][0]
                        series = np.load(self.feature_extraction+f"/features_{fts_idx}.npy")
                    else:
                        # If the extractor requires 2D input, reshape first.
                        if self.feature_extractor_requ_2d:
                            series = series.reshape(series.shape[0], -1)
                        # Transform using the feature extractor.
                        series = self.extractors[clf_idx].transform(series)
                else:
                    # No feature extraction: prepare data for the classifier.
                    if self.classifiers_requ_2d:
                        series = series.reshape(series.shape[0], -1)
                    # Otherwise, leave the series in its 3D form.
                predictions.append(
                    self.classifiers[clf_idx].predict_proba(series)
                )  
        # Send warnings if necessary
        if returned_priors:
            warn("Some time series are of insufficient length for prediction; returning prior probabilities instead.")

        return np.vstack(predictions)
    
    def _predict_past_proba(self, grouped_X, cost_matrices=None):
        """
        For each group of time series (grouped by their current length), this method computes
        predictions for all past time points up to the current length using the corresponding classifiers.
        If a time series is shorter than the smallest trained length (self.timestamps[0]),
        the class prior probabilities are returned.

        Parameters:
            grouped_X: dict
                Dictionary where keys are time series lengths and values are lists/arrays of time series.
                Each time series is assumed to have shape (D, T) and when batched, (N, D, T).
            cost_matrices: (optional) not used.

        Returns:
            A list of predictions. For each sample, the predictions are arranged over time steps.
        """
        predictions = []
        returned_priors = False
        for length, series in grouped_X.items():
            # If the series is too short, return prior probabilities.
            if length < self.timestamps[0]:
                priors = np.ones((len(series), len(self.class_prior))) * self.class_prior
                predictions.append(priors)
                returned_priors = True
            else:
                # We assume that length exactly matches one of the trained timestamps.
                ## In case the length is not in the timestamps, use the closest smaller timestamp
                #if length in self.timestamps:
                #    clf_idx = np.where(self.timestamps == length)[0][0]
                #else:
                #    # Use the closest timestamp that is less than to length (but we would need padding)
                #    clf_idx = np.where(self.timestamps < length)[0][-1]  # Last valid smaller timestamp
                clf_idx = np.where(self.timestamps == length)[0][0]
                # Ensure series is a numpy array with shape (N, D, T) where T == length.
                series = np.array(series)

                if length != self.timestamps[0]:
                    # For each time stamp up to the current one, we prepare a partial series.
                    if self.feature_extraction:
                        if os.path.isdir(str(self.feature_extraction)):
                            # Load precomputed features for each time stamp j in [0, clf_idx].
                            partial_series = [np.load(self.feature_extraction + f"/features_{j}.npy")
                                            for j in range(clf_idx + 1)]
                        else:
                            partial_series = []
                            for j in range(clf_idx + 1):
                                # Slice the series up to the j-th timestamp.
                                sliced = series[..., :self.timestamps[j]]
                                # If the extractor requires 2D, flatten the sliced data.
                                if self.feature_extractor_requ_2d:
                                    sliced = sliced.reshape(sliced.shape[0], -1)
                                transformed = self.extractors[j].transform(sliced)
                                partial_series.append(transformed)
                    else:
                        # No feature extraction: simply slice the raw series.
                        partial_series = []
                        for j in range(clf_idx + 1):
                            sliced = series[..., :self.timestamps[j]]
                            # If the classifier was trained with 2D data, flatten the slice.
                            if self.classifiers_requ_2d:
                                sliced = sliced.reshape(sliced.shape[0], -1)
                            partial_series.append(sliced)

                    # For each partial series, use the corresponding classifier to predict probabilities.
                    all_probas = [self.classifiers[j].predict_proba(x)
                                for j, x in enumerate(partial_series)]
                    # Convert the list (shape: [n_steps, N, n_classes]) into an array and
                    # transpose it to shape: (N, n_steps, n_classes)
                    all_probas = np.array(all_probas).transpose((1, 0, 2))
                    all_probas = list(all_probas)
                else:
                    # For the shortest valid series (i.e. when length == self.timestamps[0]):
                    if self.feature_extraction:
                        # Prepare input for feature extraction.
                        if self.feature_extractor_requ_2d:
                            series = series.reshape(series.shape[0], -1)

                        series = self.extractors[0].transform(series)
                    # if the classifier requires 2D, flatten the data.
                    if self.classifiers_requ_2d:
                        series = series.reshape(series.shape[0], -1)
                    # Get predictions for the first classifier.
                    all_probas = self.classifiers[0].predict_proba(series)
                    # Expand dims so that shape becomes (N, 1, n_classes)
                    all_probas = np.expand_dims(all_probas, axis=1)
                    all_probas = list(all_probas)

                predictions.extend(all_probas)

        if returned_priors:
            warn("Some time series are of insufficient length for prediction; returning prior probabilities instead.")

        return predictions


    def _predict(self, grouped_X, cost_matrices=None):
        """
        Direct prediction implementation for classifiers without predict_proba.
        Similar to _predict_proba but calls predict() instead.
        
        Parameters:
            grouped_X: dict
                Dictionary where keys are time series lengths and values are lists/arrays of time series.
            cost_matrices: optional
                Not used directly in this method.
                
        Returns:
            predictions: ndarray
                Array of predicted class indices for each sample.
        """
        predictions = []
        returned_priors = False
        for length, series in grouped_X.items():
            if length < self.timestamps[0]:
                # For shorter series, use prior probabilities to determine most likely class
                most_likely_class = np.argmax(self.class_prior)
                predictions.append(np.full(len(series), most_likely_class))
                returned_priors = True
            else:
                # Find the appropriate classifier for this length
                clf_idx = np.where(self.timestamps == length)[0][0]

                # IMPORTANT: This is the corrected timestamp-to-classifier mapping
                # if hasattr(self, "timestamp_to_idx") and length in self.timestamp_to_idx:
                #     # Use the mapping created by ECDIRE
                #     clf_idx = self.timestamp_to_idx[length]
                # else:
                #     # Use the original logic - find the closest classifier for this length
                #     clf_idx = np.searchsorted(self.timestamps, length, side='right') - 1

                series = np.array(series)
                
                # Apply feature extraction if needed
                if self.feature_extraction:
                    if os.path.isdir(str(self.feature_extraction)):
                        fts_idx = clf_idx
                        if hasattr(self, "prev_models_input_lengths"):
                            fts_idx = np.where(self.prev_models_input_lengths == length)[0][0]
                        series = np.load(self.feature_extraction+f"/features_{fts_idx}.npy")
                    else:
                        if self.feature_extractor_requ_2d:
                            series = series.reshape(series.shape[0], -1)
                        series = self.extractors[clf_idx].transform(series)
                elif self.classifiers_requ_2d:
                    series = series.reshape(series.shape[0], -1)
                    
                # Directly call predict() instead of predict_proba()
                predictions.append(self.classifiers[clf_idx].predict(series))
        # Add this at the end of _predict before returning
        if returned_priors:
            warn("Some time series are of insufficient length for prediction; using most likely class based on prior probabilities.")
        
        return np.concatenate(predictions)