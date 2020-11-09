
import logging

from sklearn.metrics import roc_auc_score

from src.bandit import BOBandit
from src.base_algorithms.classification.sklearn.decorators import additional_preprocessing_fit, \
    additional_preprocessing_predict, additional_preprocessing_get_params, additional_preprocessing_set_params
from src.utils.eval_utils import cross_validate_score, eval_performance
from src.utils.parameter_space import convert_param_space_to_config_space, convert_param_space_to_hyperopt_space

logger = logging.getLogger('BOASF')


class BaseAlgorithm(BOBandit):

    def __init__(self, **kwargs):
        super(BaseAlgorithm, self).__init__()
        self.model = None
        self.parameter_space = None
        self._model_name = None
        # self.classes_ = self.model.classes_
        self.reward = None

        self.smac_config_space = None
        self.smac_params = None

    @additional_preprocessing_fit
    def fit(self, X, y):
        if self.model is None:
            raise Exception
        self.model.fit(X, y)
        return self

    @additional_preprocessing_predict
    def predict(self, X):
        """
        Predict class labels for samples in X.

        :param X: {array-like, sparse matrix}, shape = [n_samples, n_features] Samples.
        :return:C: array,shape={n_samples} Predicted class label per sample.
        """
        if self.model is None:
            raise Exception
        return self.model.predict(X)

    @additional_preprocessing_predict
    def predict_proba(self, X):
        if self.model is None:
            raise Exception
        return self.model.predict_proba(X)

    @additional_preprocessing_predict
    def predict_log_proba(self, X):
        if self.model is None:
            raise Exception
        return self.model.predict_log_proba(X)

    def score(self, X, y, sample_weight=None):
        if self.model is None:
            raise Exception
        return self.model.score(X, y, sample_weight=sample_weight)

    def roc_auc_score(self, y_true, y_score, average=None, sample_weight=None):
        return roc_auc_score(y_true, y_score, average=average,
                             sample_weight=sample_weight)

    @additional_preprocessing_get_params
    def get_params(self, deep=True):
        if self.model is None:
            raise Exception(
                "[BaseAlgorithm] model is None, no get_params() method!")

        param_dict = self.model.get_params(deep=deep)
        new_param_dict = {}

        if self.smac_params is not None:
            # use smac_configuration_space
            # space_names = list(self.smac_params.keys())
            return self.smac_params
        else:
            space_names = self.get_configuration_space().get_space_names()

        for key in param_dict.keys():
            yes = False
            for space_name in space_names:
                if key in space_name:
                    yes = True
                    break
            if yes:
                new_param_dict[key] = param_dict[key]
        return new_param_dict

    def get_model_type(self):
        if self.model is None:
            raise Exception
        return self.model._estimator_type

    def set_config(self, params):
        """

        :param params: dict type
        :return:
        """

        for k, v in params.items():
            if not hasattr(self.model, k):
                raise TypeError(
                    "There is no attribute named %s, %s" %
                    (k, self.model))

            setattr(self, k, v)
            # print(getattr(self, k))

    @additional_preprocessing_set_params
    def set_params(self, **kwargs):
        return self.model.set_params(**kwargs)

    def get_configuration_space(self):
        raise NotImplementedError

    def set_configuration_space(self, ps=None):
        raise NotImplementedError

    def classes_(self):
        return self.model.classes_

    def new_estimator(self, config=None):
        # print("self.__class__=", self.__class__)
        # return self.__class__(**config)
        new_est = self.__class__()
        new_est.set_params(**config)
        return new_est

    def compute(self, config_id=None, config=None, budgets=None, X=None, y=None,
                X_val=None, y_val=None, metric_name=None, task='classification',
                predict_method='predict', working_directory=".", *args, **kwargs):
        model = self.new_estimator(config=config)
        assert metric_name is not None, "Evaluation rule is None, please provide a valid rule!"
        if task == 'clustering':
            model.fit(X)
            print("X.shape=", X.shape)
            y_hat = model.predict(X)
            print("y_hat.shape", y_hat.shape)
            val_score = eval_performance(metric_name=metric_name,
                                         y_score=y_hat)
        elif X_val is not None:
            model.fit(X, y)
            y_hat = model.predict(X_val)
            val_score = eval_performance(metric_name=metric_name,
                                         y_true=y_val,
                                         y_score=y_hat)
        else:
            cv_fold = kwargs['validation_strategy_args']
            assert (1 < cv_fold <= 10), "CV Fold should be: 1 < fold <= 10"
            val_score, _ = cross_validate_score(
                model, X, y, cv=cv_fold, predict_method=predict_method, metric_name=metric_name)
        self.reward = {'loss': -val_score,
                       'info': {'val_{}'.format(metric_name): val_score}}
        return self.reward

    def get_config_space(self):
        automl_config_space = self.get_configuration_space()
        cs = convert_param_space_to_config_space(automl_config_space)
        return cs

    def get_hyperopt_space(self):
        automl_config_space = self.get_configuration_space()
        return convert_param_space_to_hyperopt_space(automl_config_space)

    def print_model(self):
        print(self.model_name)

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, v):
        self._model_name = v

    def get_model(self):
        return self.model

    def __repr__(self):
        if self.model_name is not None:
            return str(self.model_name)
        return str(self.__class__.__name__)

    def get_smac_config_space(self):
        if self.smac_config_space is None:
            cs = convert_param_space_to_config_space(self.get_configuration_space())

            self.smac_config_space = cs

        return self.smac_config_space

