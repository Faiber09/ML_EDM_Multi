import numpy as np
from warnings import warn

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aeon.classification.dictionary_based._weasel_v2 import WEASELTransformerV2
from aeon.transformations.collection.feature_based import TSFresh as TSFreshFeatureExtractor
from aeon.transformations.collection.convolution_based import MiniRocket
from aeon.transformations.collection.convolution_based import HydraTransformer  # Añadido para Hydra
from aeon.classification.convolution_based._hydra import _SparseScaler  # Añadido para Hydra


class Feature_extractor:

    def __init__(self, method, scale=True, kwargs={}):

        self.method = method 
        self.scale = scale
        self.kwargs = kwargs
        self.min_length = -1 

    def fit(self, X, y=None):

        if self.method == 'minirocket':
            self.min_length = 9
            self.extractor = MiniRocket(**self.kwargs)
            if self.scale:
                self.scaler = StandardScaler(with_mean=False)
                self.extractor = make_pipeline(self.extractor, self.scaler)
        elif self.method == 'tsfresh':
            self.extractor = TSFreshFeatureExtractor(**self.kwargs)
        elif self.method == 'weasel2.0':
            self.min_length = 4
            self.extractor = WEASELTransformerV2(**self.kwargs)
        elif self.method == 'hydra':  # Añadido soporte para Hydra
            self.min_length = 9  # Similar a MiniRocket, Hydra usa kernels de longitud 9
            # Siempre incluir _SparseScaler para Hydra, independientemente del parámetro scale
            self.extractor = make_pipeline(
                HydraTransformer(**self.kwargs),
                _SparseScaler()  # Siempre usar _SparseScaler con Hydra
            )
        else:
            raise ValueError("Unknown features extraction method")

        # Bloque de pipeline eliminado de aquí, ya que ahora es específico para cada método

        is_3D = X.ndim == 3
        series_length = X.shape[-1]
        
        if series_length >= self.min_length:
            try:
                if is_3D:
                    self.extractor = self.extractor.fit(X, y).transform
                else:
                    self.extractor = self.extractor.fit(np.expand_dims(X, 1), y).transform
            except AttributeError:
                if is_3D:
                    self.extractor.fit_transform(X, y)
                else:
                    self.extractor.fit_transform(np.expand_dims(X, 1), y)
                self.extractor = self.extractor.transform
        else:
            warn(f"Time series provided are too short, (length = {series_length}) for {self.method},"
                 " using timestamps as features")
            self.extractor = self._do_nothing

        return self
    
    def _do_nothing(self, x, y=None):
        return x.squeeze()
    
    # This was modified from the original code to not expand the dimensions if series already has 3D
    def transform(self, X, y=None):
        # Check if multivariate
        is_3D = X.ndim == 3
        
        if is_3D:
            # Already in correct format
            return np.array(self.extractor(X)).reshape(len(X), -1)
        else:
            # Add channel dimension for univariate
            return np.array(self.extractor(np.expand_dims(X, 1))).reshape(len(X), -1)
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)