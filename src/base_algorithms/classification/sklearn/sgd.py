from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, \
    UnParametrizedHyperparameter
from sklearn import linear_model

from src.base_algorithms.classification.base_algorithm import BaseAlgorithm
from src.base_algorithms.classification.sklearn.decorators import additional_preprocessing_fit, merge_preprocessing_spaces, \
    merge_balancing_into_smac_space, additional_preprocessing_set_params
from src.utils.parameter_space import ParameterSpace, CategorySpace, LogFloatSpace, UniformFloatSpace


class SGDClassifier(BaseAlgorithm):
    def __init__(self,
                 loss='hinge',
                 penalty='l2',
                 alpha=0.0001,
                 l1_ratio=0.15,
                 fit_intercept=True,
                 max_iter=1000,
                 tol=1e-3,
                 shuffle=True,
                 verbose=0,
                 epsilon=0.1,
                 n_jobs=1,
                 random_state=None,
                 learning_rate='optimal',
                 eta0=0.0,
                 power_t=0.5,
                 early_stopping=False,
                 validation_fraction=0.1,
                 n_iter_no_change=5,
                 class_weight=None,
                 warm_start=False,
                 average=False):

        super(SGDClassifier, self).__init__()

        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.verbose = verbose
        self.epsilon = epsilon
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.class_weight = class_weight
        self.warm_start = warm_start
        self.average = average

        self._model_name = "SGDClassifier"
        self.model = linear_model.SGDClassifier(loss=self.loss,
                                                penalty=self.penalty,
                                                alpha=self.alpha,
                                                l1_ratio=self.l1_ratio,
                                                fit_intercept=self.fit_intercept,
                                                max_iter=self.max_iter,
                                                tol=self.tol,
                                                shuffle=self.shuffle,
                                                verbose=self.verbose,
                                                epsilon=self.epsilon,
                                                n_jobs=self.n_jobs,
                                                random_state=self.random_state,
                                                learning_rate=self.learning_rate,
                                                eta0=self.eta0,
                                                power_t=self.power_t,
                                                early_stopping=self.early_stopping,
                                                validation_fraction=self.validation_fraction,
                                                n_iter_no_change=self.n_iter_no_change,
                                                class_weight=self.class_weight,
                                                warm_start=self.warm_start,
                                                average=self.average)

    @additional_preprocessing_fit
    def fit(self, X, y, coef_init=None, intercept_init=None,
            sample_weight=None):
        self.model.fit(X, y, coef_init=coef_init, intercept_init=intercept_init,
            sample_weight=sample_weight)
        return self

    @additional_preprocessing_fit
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        if self.model is None:
            raise Exception
        self.model.partial_fit(X, y, classes=classes, sample_weight=sample_weight)
        return self

    def score(self, X, y, sample_weight=None):
        if self.model is None:
            raise Exception
        return self.model.score(X, y, sample_weight=sample_weight)

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

            loss_space = CategorySpace(name="loss",
                                       choice_space=["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
                                       default="log")
            penalty_space = CategorySpace(name="penalty", choice_space=["l1", "l2", "elasticnet", "none"], default="l2")
            alpha_space = LogFloatSpace(name="alpha", min_val=1e-7, max_val=1e-1, default=1e-4)
            l1_ratio_space = LogFloatSpace(name="l1_ratio", min_val=1e-9, max_val=1, default=0.15)
            # max_iter_space = UniformIntSpace(name="max_iter", min_val=5, max_val=20000, default=1000)
            tol_space = LogFloatSpace(name="tol", min_val=1e-5, max_val=1e-1, default=1e-4)
            epsilon_space = LogFloatSpace(name='epsilon', min_val=1e-5, max_val=1e-1, default=1e-4)

            # shuffle_space = CategorySpace(name="shuffle", choice_space=[True, False], default=True)
            learning_rate_space = CategorySpace(name="learning_rate",
                                                choice_space=["constant", "optimal", "invscaling"], default="invscaling")

            eta0_space = LogFloatSpace(name="eta0", min_val=1e-7, max_val=1e-1, default=0.01)
            power_t_space = UniformFloatSpace(name="power_t", min_val=1e-5, max_val=1, default=0.5)
            average_space = CategorySpace(name="average", choice_space=[True, False], default=False)

            parameter_space.merge([loss_space,
                                   penalty_space,
                                   alpha_space,
                                   l1_ratio_space,
                                   # max_iter_space,
                                   tol_space,
                                   # shuffle_space,
                                   epsilon_space,
                                   learning_rate_space,
                                   eta0_space,
                                   power_t_space,
                                   average_space])
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
            loss = CategoricalHyperparameter("loss",
                                             ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
                                             default_value="log")
            penalty = CategoricalHyperparameter(
                "penalty", ["l1", "l2", "elasticnet"], default_value="l2")
            alpha = UniformFloatHyperparameter(
                "alpha", 1e-7, 1e-1, log=True, default_value=0.0001)
            l1_ratio = UniformFloatHyperparameter(
                "l1_ratio", 1e-9, 1, log=True, default_value=0.15)
            fit_intercept = UnParametrizedHyperparameter("fit_intercept", "True")
            tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, log=True,
                                             default_value=1e-4)
            epsilon = UniformFloatHyperparameter(
                "epsilon", 1e-5, 1e-1, default_value=1e-4, log=True)
            learning_rate = CategoricalHyperparameter(
                "learning_rate", ["optimal", "invscaling", "constant"],
                default_value="invscaling")
            eta0 = UniformFloatHyperparameter(
                "eta0", 1e-7, 1e-1, default_value=0.01, log=True)
            power_t = UniformFloatHyperparameter("power_t", 1e-5, 1,
                                                 default_value=0.5)
            average = CategoricalHyperparameter(
                "average", [False, True], default_value=False)
            cs.add_hyperparameters([loss, penalty, alpha, l1_ratio, fit_intercept,
                                    tol, epsilon, learning_rate, eta0, power_t,
                                    average])

            # TODO add passive/aggressive here, although not properly documented?
            elasticnet = EqualsCondition(l1_ratio, penalty, "elasticnet")
            epsilon_condition = EqualsCondition(epsilon, loss, "modified_huber")

            power_t_condition = EqualsCondition(power_t, learning_rate,
                                                "invscaling")

            # eta0 is only relevant if learning_rate!='optimal' according to code
            # https://github.com/scikit-learn/scikit-learn/blob/0.19.X/sklearn/
            # linear_model/sgd_fast.pyx#L603
            eta0_in_inv_con = InCondition(eta0, learning_rate, ["invscaling",
                                                                "constant"])
            cs.add_conditions([elasticnet, epsilon_condition, power_t_condition,
                               eta0_in_inv_con])

            self.smac_config_space = cs

        return self.smac_config_space
