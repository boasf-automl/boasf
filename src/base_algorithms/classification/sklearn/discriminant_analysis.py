from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, \
    UniformIntegerHyperparameter
from sklearn import discriminant_analysis

from src.base_algorithms.classification.base_algorithm import BaseAlgorithm
from src.base_algorithms.classification.sklearn.decorators import merge_preprocessing_spaces, \
    additional_preprocessing_fit, additional_preprocessing_set_params, merge_balancing_into_smac_space
from src.utils.parameter_space import ParameterSpace, UniformFloatSpace, UniformIntSpace, CategorySpace


class LinearDiscriminantAnalysis(BaseAlgorithm):

    def __init__(self,
                 solver='svd',
                 shrinkage=None,
                 priors=None,
                 n_components=None,
                 store_covariance=False,
                 tol=0.0001):

        super(LinearDiscriminantAnalysis, self).__init__()
        self.solver = solver
        self.shrinkage = shrinkage
        self.priors = priors
        self.n_components = n_components
        self.store_covariance = store_covariance
        self.tol = tol

        self.shrinkage_factor = 0.5

        self._model_name = "LinearDA"
        self.model = discriminant_analysis.LinearDiscriminantAnalysis(solver=self.solver,
                                                                      shrinkage=self.shrinkage,
                                                                      priors=self.priors,
                                                                      n_components=self.n_components,
                                                                      store_covariance=self.store_covariance,
                                                                      tol=self.tol)

    @merge_preprocessing_spaces
    def get_configuration_space(self):
        if self.parameter_space is None:
            self.set_configuration_space()
        return self.parameter_space

    def set_configuration_space(self, ps=None):
        """

        :param ps: dict类型
        :return:
        """
        parameter_space = ParameterSpace()
        if ps is None:
            parameter_space = ParameterSpace()
            setattr(self.model, 'shrinkage_factor', 0.5)
            shrinkage_factor_space = UniformFloatSpace("shrinkage_factor", min_val=0., max_val=1., default=0.5)
            n_components_space = UniformIntSpace(name='n_components', min_val=1, max_val=251, default=10)

            # solver_space = CategorySpace(name="solver", choice_space=["svd", "lsqr", "eigen"], default="svd")
            # TODO: shrinkage parameter, possible values:
            # None: no shrinkage(default)
            # 'auto': automatic shrinkage using the Ledoit-Wolf lemma
            # float between 0 and 1: fixed shrinkage parameter.
            shrinkage_space = CategorySpace(name="shrinkage", choice_space=["auto", "manual", 'None'], default="auto")
            # tol_space = LogFloatSpace(name="tol", min_val=1e-6, max_val=1e-2, default=1e-4)
            tol_space = UniformFloatSpace(name='tol', min_val=1e-5, max_val=1e-1, default=1e-4)

            parameter_space.merge([
                # solver_space,
                shrinkage_factor_space,
                shrinkage_space,
                n_components_space,
                tol_space
            ])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space

    def decision_function(self, X):
        if self.model is None:
            raise Exception
        return self.model.decision_function(X)

    @additional_preprocessing_fit
    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def fit_transform(self, X):
        if self.model is None:
            raise Exception
        return self.model.fit_transform(X)

    def transform(self, X):
        if self.model is None:
            raise Exception
        return self.model.transform(X)

    def score(self, X, y, sample_weight=None):
        if self.model is None:
            raise Exception
        return self.model.score(X, y, sample_weight=sample_weight)

    @additional_preprocessing_set_params
    def set_params(self, **kwargs):
        if 'shrinkage_factor' not in kwargs:
            if kwargs['shrinkage'] == 'None':
                kwargs['shrinkage'] = None
            elif kwargs['shrinkage'] == 'auto':
                self.solver = 'lsqr'
                self.model.solver = 'lsqr'
            self.model.set_params(**kwargs)
        else:
            # print(kwargs)
            if kwargs['shrinkage'] == 'manual':
                kwargs['shrinkage'] = kwargs.get('shrinkage_factor')
                self.solver = 'lsqr'
                self.model.solver = 'lsqr'
            elif kwargs['shrinkage'] == 'auto':
                self.solver = 'lsqr'
                self.model.solver = 'lsqr'
            elif kwargs['shrinkage'] == 'None':
                kwargs['shrinkage'] = None
                self.solver = 'svd'
                self.model.solver = 'svd'

            del kwargs['shrinkage_factor']
            # print(kwargs)
            self.model.set_params(**kwargs)

    @merge_balancing_into_smac_space
    def get_smac_config_space(self):
        if self.smac_config_space is None:
            cs = ConfigurationSpace()
            shrinkage = CategoricalHyperparameter(
                "shrinkage", ["None", "auto", "manual"], default_value="None")
            shrinkage_factor = UniformFloatHyperparameter(
                "shrinkage_factor", 0., 1., 0.5)
            n_components = UniformIntegerHyperparameter('n_components', 1, 250, default_value=10)
            tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-4, log=True)
            cs.add_hyperparameters([shrinkage, shrinkage_factor, n_components, tol])

            cs.add_condition(EqualsCondition(shrinkage_factor, shrinkage, "manual"))

            self.smac_config_space = cs

        return self.smac_config_space


class QuadraticDiscriminantAnalysis(BaseAlgorithm):

    def __init__(self,
                 priors=None,
                 reg_param=0.0,
                 store_covariance=False,
                 tol=0.0001
                 ):

        super(QuadraticDiscriminantAnalysis, self).__init__()
        self.priors = priors
        self.reg_param = reg_param
        self.store_covariance = store_covariance
        self.tol = tol

        self._model_name = "QuadraticDA"
        self.model = discriminant_analysis.QuadraticDiscriminantAnalysis(priors=self.priors,
                                                                         reg_param=self.reg_param,
                                                                         store_covariance=self.store_covariance,
                                                                         tol=self.tol)

    @merge_preprocessing_spaces
    def get_configuration_space(self):

        if self.parameter_space is None:
            self.set_configuration_space()
        return self.parameter_space

    def set_configuration_space(self, ps=None):
        """

        :param ps: dict类型
        :return:
        """
        parameter_space = ParameterSpace()
        if ps is None:
            parameter_space = ParameterSpace()

            # priors_space = CategorySpace(choice_space=[None], default=None)
            reg_param_space = UniformFloatSpace(name="reg_param", min_val=0.0, max_val=1.0, default=0.0)
            # tol_space = LogFloatSpace(name="tol", min_val=1e-6, max_val=1e-2, default=1e-4)

            parameter_space.merge([reg_param_space,
                                   # tol_space
                                   ])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space

    def decision_function(self, X):
        if self.model is None:
            raise Exception
        return self.model.decision_function(X)

    @additional_preprocessing_fit
    def fit(self, X, y):
        if self.model is None:
            raise Exception
        self.model.fit(X, y)
        return self

    def score(self, X, y, sample_weight=None):
        if self.model is None:
            raise Exception
        return self.model.score(X, y, sample_weight=sample_weight)

    def print_model(self):
        print(self.model_name)
        return self.model_name

    @merge_balancing_into_smac_space
    def get_smac_config_space(self):
        if self.smac_config_space is None:
            cs = ConfigurationSpace()
            reg_param = UniformFloatHyperparameter('reg_param', 0.0, 1.0,
                                                   default_value=0.0)

            cs.add_hyperparameter(reg_param)

            self.smac_config_space = cs

        return self.smac_config_space


if __name__ == '__main__':
    # lda = LinearDiscriminantAnalysis()
    # import sklearn.datasets
    # X, y = sklearn.datasets.load_wine(return_X_y=True)
    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # lda.solver = 'svd'
    # lda.model.solver = 'svd'
    # lda.shrinkage = 'manual'
    # lda.model.shrinkage = 'manual'
    # lda.fit(x_train, y_train)
    # print("end")
    tmp_model = LinearDiscriminantAnalysis()
    m_param = 'shrinkage_factor'
    t = not (hasattr(tmp_model.model, m_param) or hasattr(tmp_model, m_param))
    print(t)
