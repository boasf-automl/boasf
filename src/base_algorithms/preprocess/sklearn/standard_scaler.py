from sklearn.preprocessing import StandardScaler as auto_standardscaler

from src.base_algorithms.preprocess.base_preprocessor import BasePreprocessor
from src.base_algorithms.preprocess.sklearn.decorators import numeric_data_matrix_fit, numeric_data_matrix_transform
from src.utils.parameter_space import ParameterSpace


class StandardScaler(BasePreprocessor):

    def __init__(self,
                 copy=True,
                 with_mean=True,
                 with_std=True):
        super(StandardScaler, self).__init__()
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std

        self.model = "StandardScaler"
        self.model = auto_standardscaler(copy=self.copy,
                                         with_mean=self.with_mean,
                                         with_std=self.with_std)

    @numeric_data_matrix_fit
    def fit(self, X, y=None):
        # if self.model is None:
        #     raise Exception
        self.model.fit(X, y)
        return self

    @numeric_data_matrix_transform
    def fit_transform(self, X, y=None):
        # if self.model is None:
        #     raise Exception
        return self.model.fit_transform(X, y)

    @numeric_data_matrix_transform
    def inverse_transform(self, X, copy=None):
        # if self.model is None:
        #     raise Exception
        return self.model.inverse_transform(X, copy=copy)

    @numeric_data_matrix_transform
    def transform(self, X, y='deprecated', copy=None):

        # if self.model is None:
        #     raise Exception

        return self.model.transform(X, copy=copy)

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
