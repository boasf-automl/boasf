from sklearn.preprocessing import Normalizer as auto_normalizer

from src.base_algorithms.preprocess.base_preprocessor import BasePreprocessor
from src.base_algorithms.preprocess.sklearn.decorators import numeric_data_matrix_fit, numeric_data_matrix_transform
from src.utils.parameter_space import ParameterSpace


class Normalizer(BasePreprocessor):

    def __init__(self,
                 norm="l2",
                 copy=True):

        super(Normalizer, self).__init__()
        self.norm = norm
        self.copy = copy

        self._model_name = "Normalizer"
        self.model = auto_normalizer(norm=self.norm,
                                     copy=self.copy)

    @numeric_data_matrix_fit
    def fit(self, X, y=None):
        # if self.model is None:
        #     raise Exception
        return self.model.fit(X, y=y)

    @numeric_data_matrix_transform
    def fit_transform(self, X, y=None):

        # if self.model is None:
        #     raise Exception
        return self.model.fit_transform(X, y=y)

    @numeric_data_matrix_transform
    def transform(self, X):

        # if self.model is None:
        #     raise Exception
        return self.model.transform(X)

    def get_configuration_space(self):
        if self.parameter_space is None:
            self.set_configuration_space()
        return self.parameter_space

    def set_configuration_space(self, ps=None):
        parameter_space = ParameterSpace()
        if ps is None:
            parameter_space.merge([])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space
