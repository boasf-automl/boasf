from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UnParametrizedHyperparameter, UniformFloatHyperparameter, \
    CategoricalHyperparameter

from src.base_algorithms.classification.base_algorithm import BaseAlgorithm
import sklearn
import sklearn.linear_model.passive_aggressive

from src.base_algorithms.classification.sklearn.decorators import merge_preprocessing_spaces, \
    merge_balancing_into_smac_space, additional_preprocessing_set_params
from src.utils.parameter_space import ParameterSpace, LogFloatSpace, CategorySpace


class PassiveAggressiveClassifier(BaseAlgorithm):

    def __init__(self,
                 C=1.0,
                 fit_intercept=True,
                 max_iter=1000,
                 tol=None,
                 early_stopping=False,
                 validation_fraction=0.1,
                 n_iter_no_change=5,
                 shuffle=True,
                 verbose=0,
                 loss="hinge",
                 n_jobs=None,
                 random_state=None,
                 warm_start=False,
                 class_weight=None,
                 average=False):



        super(PassiveAggressiveClassifier, self).__init__()
        self.C = C
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.shuffle = shuffle
        self.verbose = verbose
        self.loss = loss
        self.n_jobs = n_jobs
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.average = average

        self.random_state = random_state

        self._model_name = "PassiveAggressiveClassifier"
        self.model = sklearn.linear_model.passive_aggressive.PassiveAggressiveClassifier(
            C=self.C,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            shuffle=self.shuffle,
            verbose=self.verbose,
            loss=self.loss,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            warm_start=self.warm_start,
            class_weight=self.class_weight,
            average=self.average)

    def predict_proba(self, X):
        raise Exception("Algorithm {} has no method named predict_proba".format(self.model_name))

    def predict_log_proba(self, X):
        raise Exception("Algorithm {} has no method named predict_log_proba".format(self.model_name))

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
            C_space = LogFloatSpace(name='C', min_val=1e-5, max_val=10, default=1.0)
            loss_space = CategorySpace(name='loss', choice_space=["hinge", "squared_hinge"], default='hinge')
            tol_space = LogFloatSpace(name='tol', min_val=1e-5, max_val=1e-1, default=1e-4)
            avergae_space = CategorySpace(name='average', choice_space=[True, False], default=False)

            parameter_space.merge([
                C_space,
                loss_space,
                tol_space,
                avergae_space
            ])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space

    @merge_balancing_into_smac_space
    def get_smac_config_space(self):
        if self.smac_config_space is None:
            cs = ConfigurationSpace()
            C = UniformFloatHyperparameter("C", 1e-5, 10, 1.0, log=True)
            fit_intercept = UnParametrizedHyperparameter("fit_intercept", "True")
            loss = CategoricalHyperparameter(
                "loss", ["hinge", "squared_hinge"], default_value="hinge"
            )

            tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-4,
                                             log=True)
            # Note: Average could also be an Integer if > 1
            average = CategoricalHyperparameter('average', [False, True],
                                                default_value=False)
            cs.add_hyperparameters([loss, fit_intercept, tol, C, average])

            self.smac_config_space = cs

        return self.smac_config_space

    @additional_preprocessing_set_params
    def set_params(self, **kwargs):
        if 'fit_intercept' in kwargs:
            kwargs['fit_intercept'] = kwargs['fit_intercept']
        return self.model.set_params(**kwargs)
