from src.base_algorithms.classification.base_algorithm import BaseAlgorithm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import *

from src.base_algorithms.classification.sklearn.decorators import merge_preprocessing_spaces, additional_preprocessing_fit
from src.utils.parameter_space import ParameterSpace, LogFloatSpace, CategorySpace, UniformIntSpace


class GPC(BaseAlgorithm):
    def __init__(self,
                 kernel=None,
                 optimizer='fmin_l_bfgs_b',
                 n_restarts_optimizer=0,
                 max_iter_predict=100,
                 warm_start=False,
                 copy_X_train=True,
                 random_state=None,
                 multi_class='one_vs_rest',
                 n_jobs=1,
                 rbf_thetaL=1e-6,
                 rbf_thetaU=100000.0
                 ):
        super(GPC, self).__init__()
        self.kernel = kernel
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.max_iter_predict = max_iter_predict
        self.warm_start = warm_start
        self.copy_X_train = copy_X_train
        self.random_state = random_state
        self.multi_class = multi_class
        self.n_jobs = n_jobs

        self.rbf_thetaL = rbf_thetaL
        self.rbf_thetaU = rbf_thetaU

        self._model_name = "GPC"
        self.model = GaussianProcessClassifier(kernel=self.kernel,
                                               optimizer=self.optimizer,
                                               n_restarts_optimizer=self.n_restarts_optimizer,
                                               max_iter_predict=self.max_iter_predict,
                                               warm_start=self.warm_start,
                                               copy_X_train=self.copy_X_train,
                                               random_state=self.random_state,
                                               multi_class=self.multi_class,
                                               n_jobs=self.n_jobs)

    @merge_preprocessing_spaces
    def get_configuration_space(self):
        if self.parameter_space is None:
            self.set_configuration_space()
        return self.parameter_space

    def set_configuration_space(self, ps=None):
        parameter_space = ParameterSpace()
        if ps is None:
            # kernel_space = CategorySpace(name="kernel",
            #                              choice_space=[CompoundKernel,
            #                                            ConstantKernel,
            #                                            DotProduct,
            #                                            ExpSineSquared,
            #                                            Exponentiation,
            #                                            Matern,
            #                                            PairwiseKernel,
            #                                            Product,
            #                                            RBF,
            #                                            RationalQuadratic,
            #                                            Sum,
            #                                            WhiteKernel],
            #                              default=RBF(1.0))

            rbf_thetaL_space = LogFloatSpace(name='rbf_thetaL', min_val=1e-10, max_val=1e-3, default=1e-6)
            setattr(self.model, 'rbf_thetaL', 1e-6)
            rbf_thetaU_space = LogFloatSpace(name='rbf_thetaU', min_val=1.0, max_val=100000, default=100000.0)
            setattr(self.model, 'rbf_thetaU', 100000.0)

            optimizer_space = CategorySpace(name="optimizer", choice_space=["fmin_l_bfgs_b"],
                                            default="fmin_l_bfgs_b")
            n_restarts_optimizer_space = UniformIntSpace(name="n_restarts_optimizer", min_val=0, max_val=100, default=0)
            # max_iter_predict_space = UniformIntSpace(name="max_iter_predict", min_val=50, max_val=2000, default=100)
            # warm_start_space = CategorySpace(choice_space=[True, False], default=False)
            # copy_X_train_space = CategorySpace(choice_space=[True, False], default=True)
            parameter_space.merge([
                                   # kernel_space,
                                   optimizer_space,
                                   n_restarts_optimizer_space,
                                   # max_iter_predict_space
                                   ])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space

    @additional_preprocessing_fit
    def fit(self, X, y):
        n_features = X.shape[1]
        ker = RBF(length_scale=[1.0] * n_features,
                  length_scale_bounds=[(self.rbf_thetaL, self.rbf_thetaU)] * n_features)
        self.model.kernel = ker
        self.kernel = ker

        self.model.fit(X, y)
        return self

    def predict_log_proba(self, X):
        raise Exception("Algorithm {} has no method named predict_log_proba".format(self.model_name))

    def score(self, X, y, sample_weight=None):
        if self.model is None:
            raise Exception
        return self.model.score(X, y, sample_weight=sample_weight)
