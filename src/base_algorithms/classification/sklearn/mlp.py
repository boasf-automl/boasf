from sklearn.neural_network import MLPClassifier as auto_mlpclassifier

from src.base_algorithms.classification.base_algorithm import BaseAlgorithm
from src.base_algorithms.classification.sklearn.decorators import merge_preprocessing_spaces
from src.utils.parameter_space import ParameterSpace, CategorySpace


class MLPClassifier(BaseAlgorithm):

    def __init__(self,
                 hidden_layer_sizes=(100,),
                 activation='relu',
                 solver='adam',
                 alpha=0.0001,
                 batch_size='auto',
                 learning_rate='constant',
                 learning_rate_init=0.001,
                 power_t=0.5,
                 max_iter=200,
                 shuffle=True,
                 random_state=None,
                 tol=0.0001,
                 verbose=False,
                 warm_start=False,
                 momentum=0.9,
                 nesterovs_momentum=True,
                 early_stopping=False,
                 validation_fraction=0.1,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-08):
        super(MLPClassifier, self).__init__()
        self.hidden_layer_sizes=hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self._model_name = "MLPClassifier"
        self.model = auto_mlpclassifier(hidden_layer_sizes=self.hidden_layer_sizes,
                                        activation=self.activation,
                                        solver=self.solver,
                                        alpha=self.alpha,
                                        batch_size=self.batch_size,
                                        learning_rate=self.learning_rate,
                                        learning_rate_init=self.learning_rate_init,
                                        power_t=self.power_t,
                                        max_iter=self.max_iter,
                                        shuffle=self.shuffle,
                                        random_state=self.random_state,
                                        tol=self.tol,
                                        verbose=self.verbose,
                                        warm_start=self.warm_start,
                                        momentum=self.momentum,
                                        nesterovs_momentum=self.nesterovs_momentum,
                                        early_stopping=self.early_stopping,
                                        validation_fraction=self.validation_fraction,
                                        beta_1=self.beta_1,
                                        epsilon=self.epsilon
                                        )

    def score(self, X, y, sample_weight=None):
        return self.model.score(X,y,sample_weight=sample_weight)

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
            solver_space = CategorySpace(name="solver", choice_space=["lbfgs", "sgd", "adam"], default='adam')

            parameter_space.merge([solver_space])
        else:
            tmp_space = []
            for p in ps.keys():
                ps[p].set_name(p)
                tmp_space.append(ps[p])
            parameter_space.merge(tmp_space)

        self.parameter_space = parameter_space

