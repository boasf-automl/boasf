from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, CategoricalHyperparameter
from sklearn.neighbors import KNeighborsClassifier as auto_kneighborsclassifier
from sklearn.neighbors import RadiusNeighborsClassifier as auto_radiusneighborsclassifier

from src.base_algorithms.classification.base_algorithm import BaseAlgorithm
from src.base_algorithms.classification.sklearn.decorators import merge_preprocessing_spaces, additional_preprocessing_fit, \
    merge_balancing_into_smac_space
from src.utils.parameter_space import ParameterSpace, UniformIntSpace, CategorySpace


class KNeighborsClassifier(BaseAlgorithm):
    def __init__(self,
                 n_neighbors=5,
                 weights='uniform',
                 algorithm='auto',
                 leaf_size=30,
                 p=2,
                 metric='minkowski',
                 metric_params=None,
                 n_jobs=1,
                 **kwargs):
        super(KNeighborsClassifier, self).__init__()
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs

        self._model_name = 'KNeighborsClassifier'
        self.model = auto_kneighborsclassifier(n_neighbors=self.n_neighbors,
                                               weights=self.weights,
                                               algorithm=self.algorithm,
                                               leaf_size=self.leaf_size,
                                               p=self.p,
                                               metric=self.metric,
                                               metric_params=self.metric_params,
                                               n_jobs=self.n_jobs,
                                               **kwargs)

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
            n_neighbors_space = UniformIntSpace(name="n_neighbors", min_val=1, max_val=101, default=1)
            weights_space = CategorySpace(name="weights", choice_space=["uniform", "distance"], default="uniform")
            # algorithm_space = CategorySpace(name="algorithm", choice_space=["auto", "ball_tree", "kd_tree", "brute"],
            #                                 default="auto")
            # p_space = UniformIntSpace(name="p", min_val=1, max_val=5, default=2)
            p_space = CategorySpace(name="p", choice_space=[1, 2], default=2)

            parameter_space.merge([n_neighbors_space,
                                   weights_space,
                                   # algorithm_space,
                                   p_space])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space

    @additional_preprocessing_fit
    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def print_model(self):
        tmp = "KNeighborsClassifier"
        print(tmp)
        return tmp

    @merge_balancing_into_smac_space
    def get_smac_config_space(self):
        if self.smac_config_space is None:
            cs = ConfigurationSpace()
            n_neighbors = UniformIntegerHyperparameter(
                name="n_neighbors", lower=1, upper=100, log=True, default_value=1)
            weights = CategoricalHyperparameter(
                name="weights", choices=["uniform", "distance"], default_value="uniform")
            p = CategoricalHyperparameter(name="p", choices=[1, 2], default_value=2)
            cs.add_hyperparameters([n_neighbors, weights, p])

            self.smac_config_space = cs

        return self.smac_config_space


class RadiusNeighborsClassifier(BaseAlgorithm):

    def __init__(self,
                 radius=1.0,
                 weights='uniform',
                 algorithm='auto',
                 leaf_size=30,
                 p=2,
                 metric='minkowski',
                 outlier_label=None,
                 metric_params=None,
                 n_jobs=None,
                 **kwargs):
        super(RadiusNeighborsClassifier, self).__init__()
        self.radius = radius
        self.weights = weights
        self.algorithm = algorithm,
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.outlier_label = outlier_label
        self.metric_params = metric_params
        self.n_jobs = n_jobs

        self._model_name = "RadiusNeighborsClassifier"
        self.model = auto_radiusneighborsclassifier(radius=self.radius,
                                                    weights=self.weights,
                                                    algorithm=self.algorithm,
                                                    leaf_size=self.leaf_size,
                                                    p=self.p,
                                                    metric=self.metric,
                                                    outlier_label=self.outlier_label,
                                                    metric_params=self.metric_params,
                                                    n_jobs=self.n_jobs,
                                                    **kwargs)

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
            weights_space = CategorySpace(name="weights", choice_space=["uniform", "distance"], default="uniform")
            algorithm_space = CategorySpace(name="algorithm", choice_space=["ball_tree", "kd_tree", "brute", "auto"],
                                            default="auto")
            p_space = UniformIntSpace(name="p", min_val=1, max_val=5, default=2)

            parameter_space.merge([weights_space,
                                   algorithm_space,
                                   p_space])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space
