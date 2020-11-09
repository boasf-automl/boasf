from sklearn.preprocessing import QuantileTransformer as auto_quantiletransformer

from src.base_algorithms.preprocess.base_preprocessor import BasePreprocessor
from src.base_algorithms.preprocess.sklearn.decorators import numeric_data_matrix_fit, numeric_data_matrix_transform
from src.utils.parameter_space import ParameterSpace, UniformIntSpace, CategorySpace


class QuantileTransformer(BasePreprocessor):

    def __init__(self,
                 n_quantiles=1000,
                 output_distribution='uniform',
                 ignore_implicit_zeros=False,
                 subsample=int(1e5),
                 random_state=None,
                 copy=True
                 ):
        super(QuantileTransformer, self).__init__()
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.ignore_implicit_zeros = ignore_implicit_zeros
        self.subsample = subsample
        self.random_state = random_state
        self.copy = copy

        self._model_name = "QuantileTransformer"
        self.model = auto_quantiletransformer(n_quantiles=self.n_quantiles,
                                              output_distribution=self.output_distribution,
                                              ignore_implicit_zeros=self.ignore_implicit_zeros,
                                              subsample=self.subsample,
                                              random_state=self.random_state,
                                              copy=self.copy)

    @numeric_data_matrix_fit
    def fit(self, X, y=None):
        return self.model.fit(X, y=y)

    @numeric_data_matrix_transform
    def fit_transform(self, X, y=None):
        return self.model.fit_transform(X, y=y)

    @numeric_data_matrix_transform
    def inverse_transform(self, X):
        return self.model.inverse_transform(X)

    @numeric_data_matrix_transform
    def transform(self, X):
        return self.model.transform(X)

    def get_configuration_space(self):
        if self.parameter_space is None:
            self.set_configuration_space()
        return self.parameter_space

    def set_configuration_space(self, ps=None):
        parameter_space = ParameterSpace()
        if ps is None:
            n_quantiles_space = UniformIntSpace(
                name='n_quantiles', min_val=10, max_val=2000, default=1000)
            output_distribution_space = CategorySpace(
                name='output_distribution', choice_space=[
                    'uniform', 'normal'], default='uniform')

            parameter_space.merge([
                n_quantiles_space,
                output_distribution_space
            ])

        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space
