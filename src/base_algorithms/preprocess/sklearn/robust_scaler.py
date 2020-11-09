from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from sklearn.preprocessing import RobustScaler as auto_robustscaler

from src.base_algorithms.preprocess.base_preprocessor import BasePreprocessor
from src.base_algorithms.preprocess.sklearn.decorators import numeric_data_matrix_fit, numeric_data_matrix_transform
from src.utils.parameter_space import ParameterSpace


class RobustScaler(BasePreprocessor):

    """
    RobustScaler cannot be fitted to sparse inputs, but you can use the transform method on sparse inputs
    """

    def __init__(self,
                 with_centering=True,
                 with_scaling=True,
                 quantile_range=(25.0, 75.0),
                 copy=True):
        super(RobustScaler, self).__init__()
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.copy = copy

        self.q_min = 0.25
        self.q_max = 0.75

        self._model_name = "RobustScaler"
        self.model = auto_robustscaler(with_centering=self.with_centering,
                                       with_scaling=self.with_scaling,
                                       quantile_range=self.quantile_range,
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
        ps = ParameterSpace()
        ps.merge([])
        return ps

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

    def set_params(self, **params):
        import copy
        self.smac_params = copy.deepcopy(params)
        quantile_range = (0.25, 0.75)
        if params.get('q_min', None):
            quantile_range = (params.get('q_min'), params.get('q_max'))
            del params['q_min']
            del params['q_max']

        self.quantile_range = quantile_range
        self.model.quantile_range = quantile_range

    def get_smac_config_space(self):
        if self.smac_config_space is None:
            cs = ConfigurationSpace()
            q_min = UniformFloatHyperparameter(
                'q_min', 0.001, 0.3, default_value=0.25
            )
            q_max = UniformFloatHyperparameter(
                'q_max', 0.7, 0.999, default_value=0.75
            )
            cs.add_hyperparameters((q_min, q_max))

            self.smac_config_space = cs

        return self.smac_config_space