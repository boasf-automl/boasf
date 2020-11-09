from ConfigSpace.configuration_space import ConfigurationSpace

from ConfigSpace.hyperparameters import Constant, CategoricalHyperparameter, UniformFloatHyperparameter, \
    UnParametrizedHyperparameter, UniformIntegerHyperparameter
from sklearn import ensemble
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier as auto_extratreesclassifier

from src.base_algorithms.classification.base_algorithm import BaseAlgorithm
from src.base_algorithms.classification.sklearn.decorators import merge_preprocessing_spaces, additional_preprocessing_fit, \
    merge_balancing_into_smac_space, additional_preprocessing_set_params
from src.utils.parameter_space import ParameterSpace, CategorySpace, UniformFloatSpace, UniformIntSpace


class RandomForestClassifier(BaseAlgorithm):

    def __init__(self,
                 n_estimators=100,
                 criterion='gini',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features='auto',
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):

        super(RandomForestClassifier, self).__init__()

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight

        self._model_name = "RandomForestClassifier"
        self.model = ensemble.RandomForestClassifier(n_estimators=self.n_estimators,
                                                     criterion=self.criterion,
                                                     max_depth=self.max_depth,
                                                     min_samples_split=self.min_samples_split,
                                                     min_samples_leaf=self.min_samples_leaf,
                                                     min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                                     max_features=self.max_features,
                                                     max_leaf_nodes=self.max_leaf_nodes,
                                                     min_impurity_decrease=self.min_impurity_decrease,
                                                     bootstrap=self.bootstrap,
                                                     oob_score=self.oob_score,
                                                     n_jobs=self.n_jobs,
                                                     random_state=self.random_state,
                                                     verbose=self.verbose,
                                                     warm_start=self.warm_start,
                                                     class_weight=self.class_weight)
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
            # parameter_space = ParameterSpace()
            parameter_space = ParameterSpace()
            # n_estimators_space = UniformIntSpace(name="n_estimators", min_val=20, max_val=200, default=20)
            criterion_space = CategorySpace(name="criterion", choice_space=["gini", "entropy"], default="gini")
            max_features_space = UniformFloatSpace(name="max_features", min_val=0.5, max_val=1, default=1)
            # max_features_space = CategorySpace(name="max_features", choice_space=["auto", "sqrt", "log2", 1.0],
            #                                    default='auto')
            # max_depth_space
            min_samples_split_space = UniformIntSpace(name="min_samples_split", min_val=2, max_val=21, default=2)
            min_samples_leaf_space = UniformIntSpace(name="min_samples_leaf", min_val=1, max_val=21, default=1)
            # min_weight_fraction_leaf_space
            # max_leaf_nodes_space
            # Out of bag estimation only available if bootstrap=True
            bootstrap_space = CategorySpace(name="bootstrap", choice_space=[True, False], default=True)
            # oob_score_space = CategorySpace(name="oob_score", choice_space=[True, False], default=False)
            #
            # bootstrap_oobscore_relation = ConditionRelation((oob_score_space.get_name, True),
            #                                                 (bootstrap_space.get_name, True))

            # self.parameter_space.add_parameter_relation(bootstrap_oobscore_relation)

            parameter_space.merge([
                # n_estimators_space,
                criterion_space,
                max_features_space,
                min_samples_split_space,
                min_samples_leaf_space,
                bootstrap_space,
                # oob_score_space
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
        # print(f"sample_weight: {sample_weight}")
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def apply(self, X):
        if self.model is None:
            raise Exception
        return self.model.apply(X)

    def decision_path(self, X):
        if self.model is None:
            raise Exception
        return self.model.decision_path(X)

    # def new_estimator(self, config=None):
    #     model = RandomForest(**config)
    #     return model

    @merge_balancing_into_smac_space
    def get_smac_config_space(self):
        if self.smac_config_space is None:
            cs = ConfigurationSpace()
            n_estimators = Constant("n_estimators", 100)
            criterion = CategoricalHyperparameter(
                "criterion", ["gini", "entropy"], default_value="gini")

            # The maximum number of features used in the forest is calculated as m^max_features, where
            # m is the total number of features, and max_features is the hyperparameter specified below.
            # The default is 0.5, which yields sqrt(m) features as max_features in the estimator. This
            # corresponds with Geurts' heuristic.
            max_features = UniformFloatHyperparameter(
                "max_features", 0., 1., default_value=0.5)

            max_depth = UnParametrizedHyperparameter("max_depth", "None")
            min_samples_split = UniformIntegerHyperparameter(
                "min_samples_split", 2, 20, default_value=2)
            min_samples_leaf = UniformIntegerHyperparameter(
                "min_samples_leaf", 1, 20, default_value=1)
            min_weight_fraction_leaf = UnParametrizedHyperparameter("min_weight_fraction_leaf", 0.)
            max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
            min_impurity_decrease = UnParametrizedHyperparameter('min_impurity_decrease', 0.0)
            bootstrap = CategoricalHyperparameter(
                "bootstrap", ["True", "False"], default_value="True")
            cs.add_hyperparameters([n_estimators, criterion, max_features,
                                    max_depth, min_samples_split, min_samples_leaf,
                                    min_weight_fraction_leaf, max_leaf_nodes,
                                    bootstrap, min_impurity_decrease])

            self.smac_config_space = cs

        return self.smac_config_space


class DecisionTreeClassifier(BaseAlgorithm):

    def __init__(self,
                 criterion='gini',
                 splitter='best',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features=1.0,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 class_weight=None,
                 presort=False):

        super(DecisionTreeClassifier, self).__init__()

        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.presort = presort

        self._model_name = "TreeClassifier"
        self.model = tree.DecisionTreeClassifier(criterion=self.criterion,
                                                 splitter=self.splitter,
                                                 max_depth=self.max_depth,
                                                 min_samples_split=self.min_samples_split,
                                                 min_samples_leaf=self.min_samples_leaf,
                                                 min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                                 max_features=self.max_features,
                                                 random_state=self.random_state,
                                                 max_leaf_nodes=self.max_leaf_nodes,
                                                 min_impurity_decrease=self.min_impurity_decrease,
                                                 class_weight=self.class_weight,
                                                 presort=self.presort)
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

            parameter_space = ParameterSpace()

            criterion_space = CategorySpace(name="criterion", choice_space=["gini", "entropy"], default="gini")

            max_depth_space = UniformIntSpace(name="max_depth", min_val=1, max_val=12, default=5)

            # splitter_space = CategorySpace(name="splitter", choice_space=["best", "random"], default="best")
            min_samples_split_space = UniformIntSpace(name="min_samples_split", min_val=2, max_val=21, default=2)
            min_samples_leaf_space = UniformIntSpace(name="min_samples_leaf", min_val=1, max_val=21, default=1)

            # max_features_space = UniformFloatSpace(name="max_features", min_val=0.5, max_val=1, default=1)
            # max_features_space = CategorySpace(name="max_features", choice_space=["auto", "sqrt", "log2", None],
            #                                    default='auto')

            # presort_space = CategorySpace(name="presort", choice_space=[True, False], default=False)

            parameter_space.merge([criterion_space,
                                   max_depth_space,
                                   # splitter_space,
                                   min_samples_split_space,
                                   min_samples_leaf_space,
                                   # max_features_space,
                                   # presort_space
                                   ])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space

    @additional_preprocessing_fit
    def fit(self, X, y, sample_weight=None, check_input=True, X_idx_sorted=None):
        self.model.fit(X, y, sample_weight=sample_weight, check_input=check_input, X_idx_sorted=X_idx_sorted)
        return self

    def apply(self, X):
        if self.model is None:
            raise Exception
        return self.model.apply(X)

    def decision_path(self, X):
        if self.model is None:
            raise Exception
        return self.model.decision_path(X)

    # def new_estimator(self, config=None):
    #     model = DecisionTreeClassifier(**config)
    #     return model
    @merge_balancing_into_smac_space
    def get_smac_config_space(self):
        if self.smac_config_space is None:
            cs = ConfigurationSpace()
            criterion = CategoricalHyperparameter(
                "criterion", ["gini", "entropy"], default_value="gini")
            # max_depth_factor = UniformFloatHyperparameter(
            #     'max_depth_factor', 0., 2., default_value=0.5)
            min_samples_split = UniformIntegerHyperparameter(
                "min_samples_split", 2, 20, default_value=2)
            min_samples_leaf = UniformIntegerHyperparameter(
                "min_samples_leaf", 1, 20, default_value=1)
            min_weight_fraction_leaf = Constant("min_weight_fraction_leaf", 0.0)
            max_features = UnParametrizedHyperparameter('max_features', 1.0)
            max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
            min_impurity_decrease = UnParametrizedHyperparameter('min_impurity_decrease', 0.0)

            cs.add_hyperparameters([criterion, max_features,
                                    min_samples_split, min_samples_leaf,
                                    min_weight_fraction_leaf, max_leaf_nodes,
                                    min_impurity_decrease])

            self.smac_config_space = cs

        return self.smac_config_space

    @additional_preprocessing_set_params
    def set_params(self, **kwargs):
        if 'max_leaf_nodes' in kwargs:
            kwargs['max_leaf_nodes'] = None
        return self.model.set_params(**kwargs)


class ExtraTreesClassifier(BaseAlgorithm):

    def __init__(self,
                 n_estimators=100,
                 criterion='gini',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features='auto',
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super(ExtraTreesClassifier, self).__init__()
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight

        self._model_name = "ExtraTreesClassifier"
        self.model = auto_extratreesclassifier(n_estimators=self.n_estimators,
                                               criterion=self.criterion,
                                               max_depth=self.max_depth,
                                               min_samples_split=self.min_samples_split,
                                               min_samples_leaf=self.min_samples_leaf,
                                               min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                               max_features=self.max_features,
                                               max_leaf_nodes=self.max_leaf_nodes,
                                               min_impurity_decrease=self.min_impurity_decrease,
                                               bootstrap=self.bootstrap,
                                               oob_score=self.oob_score,
                                               n_jobs=self.n_jobs,
                                               random_state=self.random_state,
                                               verbose=self.verbose,
                                               warm_start=self.warm_start,
                                               class_weight=self.class_weight)

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

            # n_estimators_space = UniformIntSpace(name="n_estimators", min_val=20, max_val=200, default=20)
            criterion_space = CategorySpace(name="criterion", choice_space=["gini", "entropy"], default="gini")
            max_features_space = UniformFloatSpace(name='max_features', min_val=0., max_val=1., default=0.5)

            min_samples_split_space = UniformIntSpace(name="min_samples_split", min_val=2, max_val=21, default=2)
            min_samples_leaf_space = UniformIntSpace(name="min_samples_leaf", min_val=1, max_val=21, default=1)
            # max_features_space = CategorySpace(name="max_features", choice_space=["auto", "sqrt", "log2", None],
            #                                    default="auto")
            bootstrap_space = CategorySpace(name="bootstrap", choice_space=[True, False], default=True)

            # oob_score_space = CategorySpace(name="oob_score", choice_space=[True, False], default=False)
            #
            # bootstrap_oobscore_relation = ConditionRelation((oob_score_space.get_name, True),
            #                                                 (bootstrap_space.get_name, True))

            # self.parameter_space.add_parameter_relation(bootstrap_oobscore_relation)

            parameter_space.merge([
                # n_estimators_space,
                criterion_space,
                min_samples_split_space,
                min_samples_leaf_space,
                max_features_space,
                bootstrap_space,
                # oob_score_space
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

    def decision_path(self, X):
        if self.model is None:
            raise Exception
        return self.model.decision_path(X)

    @additional_preprocessing_fit
    def fit(self, X, y, sample_weight=None):
        if self.model is None:
            raise Exception
        return self.model.fit(X, y, sample_weight=sample_weight)

    @merge_balancing_into_smac_space
    def get_smac_config_space(self):
        if self.smac_config_space is None:
            cs = ConfigurationSpace()

            n_estimators = Constant("n_estimators", 100)
            criterion = CategoricalHyperparameter(
                "criterion", ["gini", "entropy"], default_value="gini")

            # The maximum number of features used in the forest is calculated as m^max_features, where
            # m is the total number of features, and max_features is the hyperparameter specified below.
            # The default is 0.5, which yields sqrt(m) features as max_features in the estimator. This
            # corresponds with Geurts' heuristic.
            max_features = UniformFloatHyperparameter(
                "max_features", 0., 1., default_value=0.5)

            max_depth = UnParametrizedHyperparameter(name="max_depth", value="None")

            min_samples_split = UniformIntegerHyperparameter(
                "min_samples_split", 2, 20, default_value=2)
            min_samples_leaf = UniformIntegerHyperparameter(
                "min_samples_leaf", 1, 20, default_value=1)
            min_weight_fraction_leaf = UnParametrizedHyperparameter('min_weight_fraction_leaf', 0.)
            max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
            min_impurity_decrease = UnParametrizedHyperparameter('min_impurity_decrease', 0.0)

            bootstrap = CategoricalHyperparameter(
                "bootstrap", ["True", "False"], default_value="False")
            cs.add_hyperparameters([n_estimators, criterion, max_features,
                                    max_depth, min_samples_split, min_samples_leaf,
                                    min_weight_fraction_leaf, max_leaf_nodes,
                                    min_impurity_decrease, bootstrap])

            self.smac_config_space = cs

        return self.smac_config_space

    @additional_preprocessing_set_params
    def set_params(self, **kwargs):
        if 'max_depth' in kwargs:
            kwargs['max_depth'] = None
        if 'max_leaf_nodes' in kwargs:
            kwargs['max_leaf_nodes'] = None
        return self.model.set_params(**kwargs)
