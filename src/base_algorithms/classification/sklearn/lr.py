from sklearn import linear_model

from src.base_algorithms.classification.base_algorithm import BaseAlgorithm
from src.base_algorithms.classification.sklearn.decorators import additional_preprocessing_fit, merge_preprocessing_spaces
from src.utils.parameter_space import ParameterSpace, CategorySpace, UniformFloatSpace, UniformIntSpace


class LogisticRegression(BaseAlgorithm):

    def __init__(self,
                 penalty='l2',
                 dual=False,
                 tol=1e-4,
                 C=1.0,
                 fit_intercept=True,
                 intercept_scaling=True,
                 class_weight=None,
                 random_state=None,
                 solver='lbfgs', max_iter=100,
                 multi_class='auto',
                 verbose=0,
                 warm_start=False,
                 n_jobs=1,
                 l1_ratio=None):
        super(LogisticRegression, self).__init__()
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio

        self._model_name = "LogisticRegression"
        self.model = linear_model.LogisticRegression(penalty=self.penalty,
                                                     dual=self.dual,
                                                     tol=self.tol,
                                                     C=self.C,
                                                     fit_intercept=self.fit_intercept,
                                                     intercept_scaling=self.intercept_scaling,
                                                     class_weight=self.class_weight,
                                                     random_state=self.random_state,
                                                     solver=self.solver,
                                                     max_iter=self.max_iter,
                                                     multi_class=self.multi_class,
                                                     verbose=self.verbose,
                                                     warm_start=self.warm_start,
                                                     n_jobs=self.n_jobs,
                                                     l1_ratio=self.l1_ratio)

    @additional_preprocessing_fit
    def fit(self, X, y):
        """
        Fit the model according to the given training data

        :param X: {array-like, sparse matrix}, shape(n_samples, n_features)
                    Training vector, where n_samples is the number of samples and n_features is the number of features
        :param y: array-like, shape(n_samples,)
                    Target vector relative to X.
        :return: Return it self
        """
        self.model.fit(X, y, self.class_weight)
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
            """
                parameters:

                penalty:
                    type: string, 'l1' or 'l2'
                    default: 'l2'
                    description: used to specify the norm used in the penalization.

                dual:
                    type: bool
                    default: False
                    description: Dual or primal formulation.

                tolerance:
                    type: float
                    default: 1e-4
                    description: Tolerance for stopping criteria

                C:
                    type: float
                    default: 1.0
                    description: Inverse of regularization strength; must be a positive float

                fit_intecept:
                    type: bool
                    default: True
                    description: Specifies if a constant should be added to the decision function.

                :return:

            """
            penalty_space = CategorySpace(
                name="penalty", choice_space=[
                    "l1", "l2"], default="l2")
            # dual fromulation is only implemented for l2 penalty with liblinear solver.
            # prefer dual=False when n_samples > n_features
            dual_space = CategorySpace(
                name="dual", choice_space=[False], default=False)
            # tol_space = LogFloatSpace(name="tol", min_val=1e-6, max_val=1e-2, default=1e-4)
            C_space = UniformFloatSpace(
                name="C", min_val=1e-4, max_val=1e4, default=1.0)
            # fit_intercept_space = CategorySpace([True, False], default=True)
            max_iter_space = UniformIntSpace(
                name="max_iter", min_val=50, max_val=500, default=100)
            # TODO whether set l1_ratio
            parameter_space.merge([penalty_space,
                                   dual_space,
                                   max_iter_space,
                                   # tol_space,
                                   C_space])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space

    # def new_estimator(self, config=None):
    #     return LogisticRegression(**config)


if __name__ == '__main__':
    lr = LogisticRegression()
    print(lr.model.get_params())