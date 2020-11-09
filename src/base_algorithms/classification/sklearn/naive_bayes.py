from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter
from sklearn.naive_bayes import BernoulliNB as auto_bernoulliNB
from sklearn.naive_bayes import GaussianNB as auto_gaussianNB
from sklearn.naive_bayes import MultinomialNB as auto_multinomialNB

from src.base_algorithms.classification.base_algorithm import BaseAlgorithm
from src.base_algorithms.classification.sklearn.decorators import merge_preprocessing_spaces, additional_preprocessing_fit, \
    merge_balancing_into_smac_space
from src.utils.parameter_space import ParameterSpace, UniformFloatSpace, CategorySpace, LogFloatSpace


class GaussianNB(BaseAlgorithm):
    def __init__(self,
                 priors=None,
                 var_smoothing=1e-9):
        super(GaussianNB, self).__init__()
        self.priors = priors
        self.var_smoothing = var_smoothing
        
        self._model_name = "GaussianNB"
        self.model = auto_gaussianNB(priors=self.priors,
                                     var_smoothing=self.var_smoothing)

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
            if self.model is None:
                raise Exception
            parameter_space = ParameterSpace()

        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space

    @additional_preprocessing_fit
    def fit(self, X, y, sample_weight=None):
        if self.model is None:
            raise Exception
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def partial_fit(self, X, y, sample_weight=None):
        if self.model is None:
            raise Exception
        self.model.partial_fit(X, y, sample_weight=sample_weight)
        return self

    # def predict(self, X):
    #     if self.model is None:
    #         raise Exception
    #     return self.model.predict(X)
    #
    # def predict_log_proba(self, X):
    #     if self.model is None:
    #         raise Exception
    #     return self.model.predict_log_proba(X)
    #
    # def predict_proba(self, X):
    #     if self.model is None:
    #         raise Exception
    #     return self.model.predict_proba(X)

    # def score(self, X, y, sample_weight=None):
    #     if self.model is None:
    #         raise Exception
    #     return self.model.score(X, y, sample_weight=sample_weight)

    def print(self, model):
        tmp = "GaussianNB"
        print(tmp)
        return tmp


class MultinomialNB(BaseAlgorithm):
    def __init__(self,
                 alpha=1.0,
                 fit_prior=True,
                 class_prior=None):
        super(MultinomialNB, self).__init__()
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior

        self._model_name = "MultinomialNB"
        self.model = auto_multinomialNB(alpha=self.alpha,
                                        class_prior=self.class_prior,
                                        fit_prior=self.fit_prior)

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

            alpha_space = UniformFloatSpace(name="alpha", min_val=1e-2, max_val=100, default=1.0)
            fit_prior_space = CategorySpace(name="fit_prior", choice_space=[True, False], default=True)

            parameter_space.merge([alpha_space, fit_prior_space])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space
        # print(self.parameter_space.get_space_names())

    @additional_preprocessing_fit
    def fit(self, X, y, sample_weight=None):

        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def score(self, X, y, sample_weight=None):
        if self.model is None:
            raise Exception
        return self.model.score(X, y, sample_weight=sample_weight)

    def print_model(self):
        tmp = "MultinormialNB"
        print(tmp)
        return tmp

    @merge_balancing_into_smac_space
    def get_smac_config_space(self):
        if self.smac_config_space is None:
            cs = ConfigurationSpace()

            # the smoothing parameter is a non-negative float
            # I will limit it to 100 and put it on a logarithmic scale. (SF)
            # Please adjust that, if you know a proper range, this is just a guess.
            alpha = UniformFloatHyperparameter(name="alpha", lower=1e-2, upper=100,
                                               default_value=1, log=True)

            fit_prior = CategoricalHyperparameter(name="fit_prior",
                                                  choices=["True", "False"],
                                                  default_value="True")

            cs.add_hyperparameters([alpha, fit_prior])

            self.smac_config_space = cs

        return self.smac_config_space


class BernouliNB(BaseAlgorithm):
    def __init__(self,
                 alpha=1.0,
                 binarize=0.0,
                 fit_prior=True,
                 class_prior=None):
        super(BernouliNB, self).__init__()
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.binarize = binarize
        
        self._model_name = "BernouliNB"
        self.model = auto_bernoulliNB(alpha=self.alpha,
                                      binarize=self.binarize,
                                      class_prior=self.class_prior,
                                      fit_prior=self.fit_prior)

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

            alpha_space = LogFloatSpace(name="alpha", min_val=1e-2, max_val=100, default=1.0)

            fit_prior = CategorySpace(name="fit_prior", choice_space=[True, False], default=True)

            parameter_space.merge([alpha_space, fit_prior])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space

    @additional_preprocessing_fit
    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        if self.model is None:
            raise Exception
        self.model.partial_fit(X, y)
        return self

    def score(self, X, y, sample_weight=None):
        if self.model is None:
            raise Exception
        return self.model.score(X, y, sample_weight=sample_weight)

    @merge_balancing_into_smac_space
    def get_smac_config_space(self):
        if self.smac_config_space is None:
            cs = ConfigurationSpace()

            # the smoothing parameter is a non-negative float
            # I will limit it to 1000 and put it on a logarithmic scale. (SF)
            # Please adjust that, if you know a proper range, this is just a guess.
            alpha = UniformFloatHyperparameter(name="alpha", lower=1e-2, upper=100,
                                               default_value=1, log=True)

            fit_prior = CategoricalHyperparameter(name="fit_prior",
                                                  choices=["True", "False"],
                                                  default_value="True")

            cs.add_hyperparameters([alpha, fit_prior])

            self.smac_config_space = cs

        return self.smac_config_space


if __name__=="__main__":
    import sklearn.datasets
    X, y = sklearn.datasets.load_digits(return_X_y=True)
    m = BernouliNB()
    m.fit(X, y)
    print(m.predict_proba(X))