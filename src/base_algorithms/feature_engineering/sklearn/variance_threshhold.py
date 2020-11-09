from sklearn.feature_selection import VarianceThreshold as auto_variancethreshold

from src.base_algorithms.feature_engineering.base_feature_engineer import BaseFeatureEngineer, warning_no_model
from src.utils.parameter_space import ParameterSpace


class VarianceThreshold(BaseFeatureEngineer):
    def __init__(self,
                 thresh_hold=0.0):
        super(VarianceThreshold, self).__init__()

        self.threshold = thresh_hold

        self.model = auto_variancethreshold(threshold=self.threshold)

    def fit(self, X, y=None):
        self.model = auto_variancethreshold(threshold=self.threshold)
        return self.model.fit(X, y)

    def fit_transform(self, X, y=None):
        self.model = auto_variancethreshold(threshold=self.threshold)
        return self.model.fit_transform(X, y)

    @warning_no_model
    # def get_params(self):
    #     return self._model.get_params()

    @warning_no_model
    def get_support(self):
        return self.model.get_support()

    @warning_no_model
    def inverse_transform(self, X):
        return self.model.inverse_transform(X)

    @warning_no_model
    def transform(self, X):
        return self.model.transform(X)

    # def print_model(self):
    #     tmp = "VarianceThreshold"
    #     print(tmp)
    #     return tmp

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
