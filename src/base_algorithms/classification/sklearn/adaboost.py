from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter, \
    CategoricalHyperparameter

from src.base_algorithms.classification.base_algorithm import BaseAlgorithm
from sklearn import ensemble

from src.base_algorithms.classification.sklearn.decorators import merge_preprocessing_spaces, additional_preprocessing_fit, \
    merge_balancing_into_smac_space
from src.utils.parameter_space import ParameterSpace, LogFloatSpace, UniformIntSpace, CategorySpace


class AdaBoostClassifier(BaseAlgorithm):

    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.0,
                 algorithm='SAMME',
                 random_state=None):
        super(AdaBoostClassifier, self).__init__()
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.random_state = random_state
        
        self._model_name = "AdaboostClassifier"
        self.model = ensemble.AdaBoostClassifier(base_estimator=self.base_estimator,
                                                 n_estimators=self.n_estimators,
                                                 learning_rate=self.learning_rate,
                                                 algorithm=self.algorithm,
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
            parameter_space = ParameterSpace()
            # base_estimator_space = CategorySpace(name="base_estimator",
            # choice_space=["DecisionTreeClassifier"], default="DecisionTreeClassifier")
            n_estimator_space = UniformIntSpace(name="n_estimators", min_val=50, max_val=501, default=50)
            learning_rate_space = LogFloatSpace(name="learning_rate", min_val=0.01, max_val=2, default=0.1)
            algorithm_space = CategorySpace(name="algorithm", choice_space=['SAMME', 'SAMME.R'], default='SAMME.R')
            # algorithm_space = CategorySpace(name="algorithm", choice_space=['SAMME'], default='SAMME')
            # parameter_space.add()
            parameter_space.merge([  # base_estimator_space,
                n_estimator_space,
                learning_rate_space,
                algorithm_space])
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

    def score(self, X, y, sample_weight=None):
        if self.model is None:
            raise Exception
        return self.model.score(X, y, sample_weight=sample_weight)

    @merge_balancing_into_smac_space
    def get_smac_config_space(self):
        if self.smac_config_space is None:
            cs = ConfigurationSpace()

            n_estimators = UniformIntegerHyperparameter(
                name="n_estimators", lower=50, upper=500, default_value=50, log=False)
            learning_rate = UniformFloatHyperparameter(
                name="learning_rate", lower=0.01, upper=2, default_value=0.1, log=True)
            algorithm = CategoricalHyperparameter(
                name="algorithm", choices=["SAMME.R", "SAMME"], default_value="SAMME.R")

            cs.add_hyperparameters([n_estimators, learning_rate, algorithm])

            self.smac_config_space = cs

        return self.smac_config_space


if __name__ == "__main__":
    a = AdaBoostClassifier()
    a.get_configuration_space()
    print(a.get_smac_config_space())

