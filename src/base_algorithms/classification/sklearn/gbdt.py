from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    UnParametrizedHyperparameter, CategoricalHyperparameter
from sklearn.ensemble import GradientBoostingClassifier

from src.base_algorithms.classification.base_algorithm import BaseAlgorithm
from src.base_algorithms.classification.sklearn.decorators import merge_preprocessing_spaces, \
    merge_balancing_into_smac_space, additional_preprocessing_set_params, additional_preprocessing_fit
from src.utils.parameter_space import ParameterSpace, CategorySpace, LogFloatSpace, UniformIntSpace, UniformFloatSpace


class GBDT(BaseAlgorithm):

    def __init__(self,
                 loss='deviance',
                 learning_rate=0.1,
                 n_estimators=100,
                 subsample=1.0,
                 criterion='friedman_mse',
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_depth=3,
                 min_impurity_decrease=0.0,
                 init=None,
                 random_state=None,
                 max_features=None,
                 verbose=0,
                 max_leaf_nodes=None,
                 warm_start=False,
                 presort='auto',
                 n_iter_no_change=None,
                 tol=1e-4):
        super(GBDT, self).__init__()

        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.init = init
        self.random_state = random_state
        self.max_features = max_features
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes
        self.warm_start = warm_start
        self.presort = presort
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol

        self.early_stop = None

        self._model_name = "GBDTClassifier"
        self.model = GradientBoostingClassifier(loss=self.loss,
                                                learning_rate=self.learning_rate,
                                                n_estimators=self.n_estimators,
                                                subsample=self.subsample,
                                                criterion=self.criterion,
                                                min_samples_split=self.min_samples_split,
                                                min_samples_leaf=self.min_samples_leaf,
                                                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                                max_depth=self.max_depth,
                                                min_impurity_decrease=self.min_impurity_decrease,
                                                init=self.init,
                                                random_state=self.random_state,
                                                max_features=self.max_features,
                                                verbose=self.verbose,
                                                max_leaf_nodes=self.max_leaf_nodes,
                                                warm_start=self.warm_start,
                                                presort=self.presort,
                                                n_iter_no_change=self.n_iter_no_change,
                                                tol=self.tol)
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

            loss_space = CategorySpace(name="loss", choice_space=['deviance', 'exponential'], default='deviance')
            learning_rate_space = LogFloatSpace(name="learning_rate", min_val=0.01, max_val=1, default=0.1)
            n_estimators_space = UniformIntSpace(name="n_estimators", min_val=50, max_val=500, default=100)
            max_depth_space = UniformIntSpace(name='max_depth', min_val=1, max_val=11, default=3)

            criterion_space = CategorySpace(name="criterion", choice_space=["friedman_mse", "mse", "mae"],
                                            default="friedman_mse")
            min_samples_split_space = UniformIntSpace(name="min_samples_split", min_val=2, max_val=21, default=2)
            min_samples_leaf_space = UniformIntSpace(name="min_samples_leaf", min_val=1, max_val=21, default=1)
            subsample_space = UniformFloatSpace(name='subsample', min_val=0.01, max_val=1.0, default=1.0)
            max_features_space = UniformFloatSpace(name='max_features', min_val=0.1, max_val=1.0, default=1)
            # max_features_space = CategorySpace(name="max_features", choice_space=["auto", "sqrt", "log2", 1.0],
            #                                    default='auto')

            parameter_space.merge([
                loss_space,
                learning_rate_space,
                n_estimators_space,
                max_depth_space,
                criterion_space,
                min_samples_leaf_space,
                min_samples_split_space,
                subsample_space,
                max_features_space
            ])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space

    def apply(self, X):
        if self.model is None:
            raise Exception
        return self.model.apply(X)

    def decision_function(self, X):
        if self.model is None:
            raise Exception
        return self.model.decision_function(X)

    @additional_preprocessing_fit
    def fit(self, X, y, sample_weight=None):
        if self.model is None:
            raise Exception
        return self.model.fit(X, y, sample_weight=sample_weight)

    # def new_estimator(self, config=None):
    #     model = GBDT(**config)
    #     return model

    def print_model(self):
        print(self.model_name)
        return self.model_name

    @merge_balancing_into_smac_space
    def get_smac_config_space(self):
        if self.smac_config_space is None:
            cs = ConfigurationSpace()

            loss = CategoricalHyperparameter("loss", ['deviance', 'exponential'], "deviance")
            learning_rate = UniformFloatHyperparameter(
                name="learning_rate", lower=0.01, upper=1, default_value=0.1, log=True)
            # max_iter = UniformIntegerHyperparameter(
            #     "max_iter", 32, 512, default_value=100)
            min_samples_leaf = UniformIntegerHyperparameter(
                name="min_samples_leaf", lower=1, upper=200, default_value=20, log=True)
            max_depth = UnParametrizedHyperparameter(
                name="max_depth", value="None")
            max_leaf_nodes = UniformIntegerHyperparameter(
                name="max_leaf_nodes", lower=3, upper=2047, default_value=31, log=True)
            # max_bins = Constant("max_bins", 256)
            # l2_regularization = UniformFloatHyperparameter(
            #     name="l2_regularization", lower=1E-10, upper=1, default_value=1E-10, log=True)
            # early_stop = CategoricalHyperparameter(
            #     name="early_stop", choices=["off", "train", "valid"], default_value="off")
            tol = UnParametrizedHyperparameter(
                name="tol", value=1e-7)
            # scoring = UnParametrizedHyperparameter(
            #     name="scoring", value="loss")
            n_iter_no_change = UniformIntegerHyperparameter(
                name="n_iter_no_change", lower=1, upper=20, default_value=10)
            # validation_fraction = UniformFloatHyperparameter(
            #     name="validation_fraction", lower=0.01, upper=0.4, default_value=0.1)

            # cs.add_hyperparameters([loss, learning_rate, max_iter, min_samples_leaf,
            #                         max_depth, max_leaf_nodes, max_bins, l2_regularization,
            #                         early_stop, tol, scoring, n_iter_no_change,
            #                         validation_fraction])

            cs.add_hyperparameters([loss, learning_rate, min_samples_leaf,
                                    max_depth, max_leaf_nodes,
                                    tol, n_iter_no_change,
                                    ])

            # n_iter_no_change_cond = InCondition(
            #     n_iter_no_change, early_stop, ["valid", "train"])
            # validation_fraction_cond = EqualsCondition(
            #     validation_fraction, early_stop, "valid")
            #
            # cs.add_conditions([n_iter_no_change_cond, validation_fraction_cond])

            self.smac_config_space = cs

        return self.smac_config_space

    @additional_preprocessing_set_params
    def set_params(self, **kwargs):
        if 'early_stop' in kwargs:
            self.early_stop = kwargs['early_stop']

        return self.model.set_params(**kwargs)

