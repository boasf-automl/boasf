from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.forbidden import ForbiddenAndConjunction, ForbiddenEqualsClause
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter, \
    UniformIntegerHyperparameter, UnParametrizedHyperparameter, Constant
from sklearn.svm import LinearSVC as auto_linearsvc
from sklearn.svm import NuSVC as auto_nusvc
from sklearn.svm import SVC as auto_svc

from src.base_algorithms.classification.base_algorithm import BaseAlgorithm
from src.base_algorithms.classification.sklearn.decorators import merge_preprocessing_spaces, additional_preprocessing_fit, \
    merge_balancing_into_smac_space, additional_preprocessing_set_params
from src.utils.parameter_space import ParameterSpace, LogFloatSpace, CategorySpace, UniformIntSpace, UniformFloatSpace


class SVC(BaseAlgorithm):

    """
    the fit time complexity is more than quadratic with the number of samples which makes it hard to scale to dataset
    with more than a couple of 10000 samples
    """

    def __init__(self,
                 C=1.0,
                 kernel='rbf',
                 degree=3,
                 gamma='auto',
                 coef0=0.0,
                 shrinking=True,
                 probability=False,
                 tol=0.001,
                 cache_size=200,
                 class_weight=None,
                 verbose=False,
                 max_iter=-1,
                 decision_function_shape='ovr',
                 random_state=None):
        super(SVC, self).__init__()
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.random_state = random_state
        
        self._model_name = "SVC"
        self.model = auto_svc(C=self.C,
                              kernel=self.kernel,
                              degree=self.degree,
                              gamma=self.gamma,
                              coef0=self.coef0,
                              shrinking=self.shrinking,
                              probability=self.probability,
                              tol=self.tol,
                              cache_size=self.cache_size,
                              class_weight=self.class_weight,
                              verbose=self.verbose,
                              max_iter=self.max_iter,
                              decision_function_shape=self.decision_function_shape,
                              random_state=self.random_state)
        self.reward = None

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
            C_space = LogFloatSpace(name="C", min_val=0.03125, max_val=32768, default=1.0)
            kernel_space = CategorySpace(name="kernel", choice_space=[ "poly", "rbf", "sigmoid"],
                                         default="rbf")
            degree_space = UniformIntSpace(name="degree", min_val=1, max_val=6, default=3)
            gamma_space = LogFloatSpace(name="gamma", min_val=3.0517578125e-05, max_val=8, default=0.1)

            coef0_space = UniformFloatSpace(name="coef0", min_val=-1, max_val=1, default=0)
            # probability_space = CategorySpace(name="probability", choice_space=[True, False], default=False)
            shrinking_space = CategorySpace(name="shrinking", choice_space=[True, False], default=True)
            tol_space = LogFloatSpace(name="tol", min_val=1e-5, max_val=1e-1, default=1e-3)

            parameter_space.merge([C_space,
                                   kernel_space,
                                   degree_space,
                                   gamma_space,
                                   coef0_space,
                                   # probability_space,
                                   shrinking_space,
                                   tol_space
                                   ])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space

    @additional_preprocessing_fit
    def fit(self, X, y, sample_weight=None):
        return self.model.fit(X, y, sample_weight=sample_weight)

    # def predict(self, X):
    #     if self.model is None:
    #         raise Exception
    #     return self.model.predict(X)
    #
    # def predict_proba(self, X):
    #     return self.model.predict_proba(X)
    #
    # def score(self, X, y, sample_weight=None):
    #     if self.model is None:
    #         raise Exception
    #     return self.model.score(X, y, sample_weight=sample_weight)

    # def new_estimator(self, config=None):
    #     model = SVC(**config)
    #     return model
    @merge_balancing_into_smac_space
    def get_smac_config_space(self):
        if self.smac_config_space is None:
            cs = ConfigurationSpace()
            C = UniformFloatHyperparameter("C", 0.03125, 32768, log=True,
                                           default_value=1.0)
            # No linear kernel here, because we have liblinear
            kernel = CategoricalHyperparameter(name="kernel",
                                               choices=["rbf", "poly", "sigmoid"],
                                               default_value="rbf")
            degree = UniformIntegerHyperparameter("degree", 2, 5, default_value=3)
            gamma = UniformFloatHyperparameter("gamma", 3.0517578125e-05, 8,
                                               log=True, default_value=0.1)
            # TODO this is totally ad-hoc
            coef0 = UniformFloatHyperparameter("coef0", -1, 1, default_value=0)
            # probability is no hyperparameter, but an argument to the SVM algo
            shrinking = CategoricalHyperparameter("shrinking", [True, False],
                                                  default_value=True)
            tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-3,
                                             log=True)
            # cache size is not a hyperparameter, but an argument to the program!
            max_iter = UnParametrizedHyperparameter("max_iter", -1)

            cs.add_hyperparameters([C, kernel, degree, gamma, coef0, shrinking,
                                    tol, max_iter])

            degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
            coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])
            cs.add_condition(degree_depends_on_poly)
            cs.add_condition(coef0_condition)

            self.smac_config_space = cs

        return self.smac_config_space


class NuSVC(BaseAlgorithm):

    def __init__(self,
                 nu=0.5,
                 kernel='rbf',
                 degree=3,
                 gamma='auto',
                 coef0=0.0,
                 shrinking=True,
                 probability=False,
                 tol=0.001,
                 cache_size=200,
                 class_weight=None,
                 verbose=False,
                 max_iter=-1,
                 decision_function_shape='ovr',
                 random_state=None):
        super(NuSVC, self).__init__()
        self.nu = nu
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.random_state = random_state
        
        self._model_name = "NuSVC"
        self.model = auto_nusvc(nu=self.nu,
                                kernel=self.kernel,
                                degree=self.degree,
                                gamma=self.gamma,
                                coef0=self.coef0,
                                shrinking=self.shrinking,
                                probability=self.probability,
                                tol=self.tol,
                                cache_size=self.cache_size,
                                class_weight=self.class_weight,
                                verbose=self.verbose,
                                max_iter=self.max_iter,
                                decision_function_shape=self.decision_function_shape,
                                random_state=self.random_state)

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
            # nu_space = UniformFloatSpace(name="nu", min_val=1e-9, max_val=1, default=0.5)
            kernel_space = CategorySpace(name="kernel", choice_space=["linear", "poly", "rbf", "sigmoid"],
                                         default="rbf")
            degree_space = UniformIntSpace(name="degree", min_val=1, max_val=5, default=3)
            gamma_space = LogFloatSpace(name="gamma", min_val=1e-5, max_val=1, default=0.1)
            coef0_space = UniformFloatSpace(name="coef0", min_val=-1, max_val=1, default=0)
            probability_space = CategorySpace(name="probability", choice_space=[True, False], default=False)
            shrinking_space = CategorySpace(name="shrinking", choice_space=[True, False], default=True)
            tol_space = LogFloatSpace(name="tol", min_val=1e-6, max_val=1e-2, default=1e-3)

            parameter_space.merge([
                kernel_space,
                degree_space,
                gamma_space,
                coef0_space,
                probability_space,
                shrinking_space,
            ])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space

    @additional_preprocessing_fit
    def fit(self, X, y, sample_weight=None):
        self.model = auto_nusvc(nu=self.nu,
                                kernel=self.kernel,
                                degree=self.degree,
                                gamma=self.gamma,
                                coef0=self.coef0,
                                shrinking=self.shrinking,
                                probability=self.probability,
                                tol=self.tol,
                                cache_size=self.cache_size,
                                class_weight=self.class_weight,
                                verbose=self.verbose,
                                max_iter=self.max_iter,
                                decision_function_shape=self.decision_function_shape,
                                random_state=self.random_state)
        return self.model.fit(X, y, sample_weight=sample_weight)

    # def predict(self, X):
    #     if self.model is None:
    #         raise Exception
    #     return self.model.predict(X)
    #
    # def score(self, X, y, sample_weight=None):
    #     if self.model is None:
    #         raise Exception
    #     return self.model.score(X, y, sample_weight=sample_weight)


class LinearSVC(BaseAlgorithm):

    def __init__(self,
                 penalty='l2',
                 loss='squared_hinge',
                 dual=False,
                 tol=1e-4,
                 C=1.0,
                 multi_class='ovr',
                 fit_intercept=True,
                 intercept_scaling=1,
                 class_weight=None,
                 verbose=0,
                 random_state=None,
                 max_iter=1000):

        super(LinearSVC, self).__init__()

        self.penalty = penalty
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.C = C
        self.multi_class = multi_class
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.verbose = verbose
        self.random_state = random_state
        self.max_iter = max_iter

        self._model_name = "LinearSVC"
        self.model = auto_linearsvc(penalty=self.penalty,
                                    loss=self.loss,
                                    dual=self.dual,
                                    tol=self.tol,
                                    C=self.C,
                                    multi_class=self.multi_class,
                                    fit_intercept=self.fit_intercept,
                                    intercept_scaling=self.intercept_scaling,
                                    class_weight=self.class_weight,
                                    verbose=self.verbose,
                                    random_state=self.random_state,
                                    max_iter=self.max_iter)

    @additional_preprocessing_fit
    def fit(self, X, y, sample_weight=None):
        self.model = auto_linearsvc(penalty=self.penalty,
                                    loss=self.loss,
                                    dual=self.dual,
                                    tol=self.tol,
                                    C=self.C,
                                    multi_class=self.multi_class,
                                    fit_intercept=self.fit_intercept,
                                    intercept_scaling=self.intercept_scaling,
                                    class_weight=self.class_weight,
                                    verbose=self.verbose,
                                    random_state=self.random_state,
                                    max_iter=self.max_iter)
        return self.model.fit(X, y, sample_weight=sample_weight)

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

            penalty_space = CategorySpace(name="penalty", choice_space=["l1", "l2"], default="l2")
            loss_space = CategorySpace(name="loss", choice_space=["hinge", "squared_hinge"], default="squared_hinge")

            tol_space = LogFloatSpace(name='tol', min_val=1e-5, max_val=1e-1, default=1e-4)

            C_space = LogFloatSpace(name="C", min_val=0.03125, max_val=32768, default=1.0)

            parameter_space.merge([penalty_space,
                                   tol_space,
                                   loss_space,
                                   C_space
                                   ])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space

    # def score(self, X, y, sample_weight=None):
    #     if self.model is None:
    #         raise Exception
    #     return self.model.score(X, y, sample_weight=sample_weight)

    @merge_balancing_into_smac_space
    def get_smac_config_space(self):
        if self.smac_config_space is None:
            cs = ConfigurationSpace()
            penalty = CategoricalHyperparameter(
                "penalty", ["l1", "l2"], default_value="l2")
            loss = CategoricalHyperparameter(
                "loss", ["hinge", "squared_hinge"], default_value="squared_hinge")
            dual = Constant("dual", "False")
            # This is set ad-hoc
            tol = UniformFloatHyperparameter(
                "tol", 1e-5, 1e-1, default_value=1e-4, log=True)
            C = UniformFloatHyperparameter(
                "C", 0.03125, 32768, log=True, default_value=1.0)
            multi_class = Constant("multi_class", "ovr")
            # These are set ad-hoc
            fit_intercept = Constant("fit_intercept", "True")
            intercept_scaling = Constant("intercept_scaling", 1)
            cs.add_hyperparameters([penalty, loss, dual, tol, C, multi_class,
                                    fit_intercept, intercept_scaling])

            penalty_and_loss = ForbiddenAndConjunction(
                ForbiddenEqualsClause(penalty, "l1"),
                ForbiddenEqualsClause(loss, "hinge")
            )
            constant_penalty_and_loss = ForbiddenAndConjunction(
                ForbiddenEqualsClause(dual, "False"),
                ForbiddenEqualsClause(penalty, "l2"),
                ForbiddenEqualsClause(loss, "hinge")
            )
            penalty_and_dual = ForbiddenAndConjunction(
                ForbiddenEqualsClause(dual, "False"),
                ForbiddenEqualsClause(penalty, "l1")
            )
            cs.add_forbidden_clause(penalty_and_loss)
            cs.add_forbidden_clause(constant_penalty_and_loss)
            cs.add_forbidden_clause(penalty_and_dual)

            self.smac_config_space = cs

        return self.smac_config_space

    @additional_preprocessing_set_params
    def set_params(self, **kwargs):
        if 'dual' in kwargs:
            kwargs['dual'] = kwargs['dual']  # False

        return self.model.set_params(**kwargs)