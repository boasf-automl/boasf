from sklearn.preprocessing import MaxAbsScaler as auto_maxabsscaler

from src.base_algorithms.preprocess.base_preprocessor import BasePreprocessor
from src.base_algorithms.preprocess.sklearn.decorators import numeric_data_matrix_transform, numeric_data_matrix_fit
from src.utils.parameter_space import ParameterSpace


class MaxAbsScaler(BasePreprocessor):

    def __init__(self,
                 copy=True):

        super(MaxAbsScaler, self).__init__()
        self.copy = copy

        self._model_name = "MaxAbsScaler"
        self.model = auto_maxabsscaler(copy=self.copy)

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
    def inverse_transform(self, X):
        # if self.model is None:
        #     raise Exception
        return self.model.inverse_transform(X)

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
