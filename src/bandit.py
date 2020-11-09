#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2020-08-14
import copy
import logging
import math
import multiprocessing
import time
from functools import partial

from hyperopt import Trials, STATUS_FAIL, STATUS_OK, fmin, tpe, space_eval, hp
import numpy as np

from src.utils.eval_utils import cross_validate_score
from src.utils.parameter_space import CategorySpace

logger = logging.getLogger('BOASF')


def generate_orthogonal_bandits(splited_spaces, model_name, depth, bandits, tmp_spa, gucb_c=2):
    if depth == len(splited_spaces):
        from src.base_algorithms.get_algorithm import get_algorithm_by_key
        tmp_model = get_algorithm_by_key(model_name)
        tmp_model.gucb_c = gucb_c
        tmp_model.set_configuration_space(tmp_spa)
        bandits[len(bandits)] = tmp_model
        return
    for i in range(len(splited_spaces[depth])):
        tmp_spa[splited_spaces[depth][i].get_name()] = splited_spaces[depth][i]
        generate_orthogonal_bandits(splited_spaces, model_name, depth + 1, bandits, tmp_spa, gucb_c)
        del tmp_spa[splited_spaces[depth][i].get_name()]


class BOBandit(object):
    def __init__(self,
                 budget_type="time",
                 gucb_c=2):
        self._trials = Trials()  # record all trials in this bandit
        self._u = 0  # mean
        self._v = 1e-5  # variance
        self._gucb_c = gucb_c  # constant to control gaussian UCB
        self._budget_type = budget_type
        self._num_of_action = 0
        self._worst_score = 0
        self._best_score = []

        self._model_name = None

        # record
        self._record = []
        self._length_of_each_added_record = []
        self._model_parameters = []  # have final model parameters
        self._pipeline_parameters = []  # have parameters: model_name, balancing, rescaling, variance_threshold

        self._balancing_space = CategorySpace(name="balancing", choice_space=[True, False], default=False)
        self._rescaling_space = CategorySpace(name="rescaling",
                                              choice_space=["MinMaxScaler", "None", "Normalizer",
                                                            "QuantileTransformer", "RobustScaler", "StandardScaler"],
                                              default="None")

        self._variance_threshold_space = CategorySpace(name="variance_threshold", choice_space=[True, False],
                                                       default=False)

        self._preprocessing_spaces = [self._balancing_space, self._rescaling_space, self._variance_threshold_space]

        self._balancing = False
        self._rescaling = "None"
        self._variance_threshold = False

        self._preprocessing_models = []

        self._has_mergered_param_sapce = False

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, v):
        self._model_name = v

    @property
    def balancing(self):
        return self._balancing

    @balancing.setter
    def balancing(self, val):
        assert val in self._balancing_space.get_choice_space(), "val {} is illegal."
        self._balancing = val

    @property
    def rescaling(self):
        return self._rescaling

    @rescaling.setter
    def rescaling(self, val):
        assert val in self._rescaling_space.get_choice_space(), "val {} is illegal"
        self._rescaling = val

    @property
    def variance_threshold(self):
        return self._variance_threshold

    @variance_threshold.setter
    def variance_threshold(self, val):
        assert val in self._variance_threshold_space.get_choice_space(), "val {} is illegal"
        self._variance_threshold = val

    @property
    def preprocessing_spaces(self):
        return self._preprocessing_spaces

    @preprocessing_spaces.setter
    def preprocessing_spaces(self, val):
        self._preprocessing_spaces = val

    def disable_preprocessing(self):
        """
        Don't use preprocessing algorithms.
        """
        self.preprocessing_spaces = []

    def get_mean(self):
        if len(self._record) > 0:
            self._u = np.mean(self._record)
        return self._u

    def get_variation(self):
        if len(self._record) > 0:
            self._v = np.std(self._record)
        return self._v

    def get_num_of_action(self):
        return self._num_of_action

    def _gaussian_ucb_score(self):
        # TODO in paper should be sqrt(N)
        return self.get_mean() + self._gucb_c * self.get_variation() / math.log(self._num_of_action + 2, 2)

    def _max_record(self):
        if len(self._trials.trials) < 1:
            return self._worst_score
        else:
            return max(self._record)

    def get_ucb_score(self):
        return self._gaussian_ucb_score()

    def add_record(self, record, model_parameters=None, pipeline_parameters=None):
        if model_parameters is not None:
            self._record.extend(record)
            self._num_of_action += len(record)

            self._length_of_each_added_record.append(len(record))

            self._model_parameters.extend(model_parameters)
            self._pipeline_parameters.extend(pipeline_parameters)

    def get_hyperopt_space(self):
        """
        Should be implemented in child class.
        """
        raise NotImplementedError()

    def _run_trials(self, X, y, input_dict, return_dict, random_state, task="classification",
                    cv_fold=3,
                    metric_name="balanced_accuracy_score"):
        model_space = {'model_name': hp.choice('model_name', [self._model_name])}
        param_space = self.get_hyperopt_space()  # include balancing, rescaling, variance_threshold

        space = {**model_space, **param_space}

        trials_step = 1
        total_trials = input_dict["total_trials"]
        max_trials = len(total_trials) + trials_step

        def objective(params):
            from src.base_algorithms.get_algorithm import get_algorithm_by_key
            model_name = params['model_name']
            model = get_algorithm_by_key(model_name)

            pipeline_params_keys = ['balancing', 'rescaling', 'variance_threshold']
            pipeline_params = dict([(k, params[k]) for k in pipeline_params_keys if k in params])
            pipeline_params['model_name'] = model_name
            model_params = dict([(k, params[k]) for k in params.keys() if k not in pipeline_params_keys])

            model.set_params(**params)

            logger.info(f"model_name: {model_name}; "
                        f"sample model params: {model_params}; "
                        f"sample pipeline params: {pipeline_params}")

            try:
                val_score, _ = cross_validate_score(model, X, y,
                                                    cv=cv_fold,
                                                    random_state=random_state,
                                                    metric_name=metric_name)
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(traceback.format_exc())
                return {'loss': 0, 'status': STATUS_FAIL, 'model_params': model_params,
                        'pipeline_params': pipeline_params}

            return {'loss': -val_score, 'status': STATUS_OK, 'model_params': model_params,
                    'pipeline_params': pipeline_params}

        logger.info("Before space_eval; len(total_trials)={}".format(len(total_trials)))
        try:
            best = fmin(fn=objective,
                        space=space,
                        # algo=partial(tpe.suggest, n_startup_jobs=len(space)),  # random sample n+1 times (n is number of hyperpameters)
                        algo=partial(tpe.suggest, n_startup_jobs=20),  # test
                        max_evals=max_trials,
                        trials=total_trials,
                        rstate=random_state
                        )
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(traceback.format_exc())
            return
        best_params = space_eval(space, best)

        val_score = -total_trials.losses()[-1]
        last_mdoel_params = total_trials.results[-1]['model_params']
        last_pipeline_params = total_trials.results[-1]['pipeline_params']

        logger.info(f"After space_eval; len(total_trials)={len(total_trials)}; "
                    f"best_score = {-total_trials.best_trial['result']['loss']}; "
                    f"best params={best_params}; "
                    f"last val_score {val_score}; "
                    f"last model params {last_mdoel_params}; "
                    f"last pipeline params {last_pipeline_params}")

        # logger.info("total_trials losses: {}".format(total_trials.losses()))
        # logger.info("total_trials results: {}".format(total_trials.results))

        return_dict["total_trials"] = total_trials
        return_dict["val_score"] = val_score
        return_dict["model_params"] = last_mdoel_params
        return_dict["pipeline_params"] = last_pipeline_params

    def compute_boasf(self, X, y,
                      resource=3, resource_type="time",
                      metric_name="balanced_accuracy_score",
                      task="classification",
                      cv_fold=3,
                      per_model_time_budget=60.0,
                      random_state=None):

        assert metric_name is not None, "Evaluation rule is None, please provide a valid rule!"
        res = []
        model_parameters = []
        pipeline_parameters = []

        if resource_type == "round":
            raise NotImplementedError
        else:
            tmp_start_time = time.time()

            print("start pull bandit")

            while True:
                mgr = multiprocessing.Manager()
                return_dict = mgr.dict()
                input_dict = mgr.dict()
                input_dict["total_trials"] = self._trials
                my_rstate = np.random.RandomState(random_state)
                try:
                    return_dict["val_score"] = None
                    return_dict["total_trials"] = None
                    assert (1 < cv_fold <= 10), "CV Fold should be: 1 < fold <= 10"

                    p = multiprocessing.Process(target=self._run_trials,
                                                args=(X, y, input_dict, return_dict, my_rstate, task, cv_fold,
                                                      metric_name))
                    p.start()
                    time_limit = min(per_model_time_budget, resource - (time.time() - tmp_start_time))
                    p.join(time_limit)
                    if p.is_alive():
                        p.terminate()

                    if return_dict["total_trials"] is not None and return_dict["val_score"] is not None:
                        self._trials = return_dict["total_trials"]
                        res.append(return_dict["val_score"])
                        model_parameters.append(return_dict["model_params"])
                        pipeline_parameters.append(return_dict["pipeline_params"])
                    else:
                        logger.info(f"time is not enough for bandit")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    logger.error(traceback.format_exc())
                if time.time() - tmp_start_time >= resource:
                    break

            logger.info("end pull bandit")

        return res, model_parameters, pipeline_parameters

    def get_best_record(self):
        return self._max_record()

    def get_min_record(self):
        if len(self._record) < 1:
            return -1
        return min(self._record)

    def get_best_model_parameters(self):
        if len(self._record) < 1:
            return None
        # print("len_para", len(self._model_parameters), len(self._record))
        # print("para", self._record)
        ind = np.argmax(np.array(self._record))
        # print("para ind", ind, self._record[ind])
        return self._model_parameters[ind]

    def get_best_pipeline_parameters(self):
        if len(self._record) < 1:
            return None

        ind = np.argmax(np.array(self._record))

        return self._pipeline_parameters[ind]

    def print_statistic(self):
        info_0 = "[model_name]: {}\n".format(self._model_name)
        info_1 = "trials.length: {}\n".format(len(self._trials.trials))
        info_2 = "num_of_action: {}\n".format(self._num_of_action)
        info_3 = "record length: {}\n".format(len(self._record))
        info_4 = "record mean: {}\n".format(self._u)
        info_5 = "record variance: {}\n".format(self._v)
        info_6 = "record max: {}\n".format(self.get_best_record())
        info_7 = "record min: {}\n".format(self.get_min_record())
        info_8 = "length of every added records: {}\n".format(self._length_of_each_added_record)
        # print(info_1)
        # print(info_2)
        # print(info_3)
        # print(info_4)
        # print(info_5)
        # print(info_6)
        # print(info_7)
        # print(info_8)
        logger.info(info_0)
        logger.info(info_1)
        logger.info(info_2)
        logger.info(info_3)
        logger.info(info_4)
        logger.info(info_5)
        logger.info(info_6)
        logger.info(info_7)
        logger.info(info_8)

        return info_0 + info_1 + info_2 + info_3 + info_4 + info_5 + info_6 + info_7 + info_8
