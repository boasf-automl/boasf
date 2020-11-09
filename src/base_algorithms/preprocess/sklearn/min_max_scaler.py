from sklearn.preprocessing import MinMaxScaler as auto_minmaxscaler

from src.base_algorithms.preprocess.base_preprocessor import BasePreprocessor
from src.base_algorithms.preprocess.sklearn.decorators import numeric_data_matrix_fit, numeric_data_matrix_transform
from src.utils.parameter_space import ParameterSpace


class MinMaxScaler(BasePreprocessor):

    def __init__(self,
                 feature_range=(0, 1),
                 copy=True):

        super(MinMaxScaler, self).__init__()
        self.feature_range = feature_range
        self.copy = copy

        self.model = "MinMaxScaler"
        self.model = auto_minmaxscaler(feature_range=self.feature_range,
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
