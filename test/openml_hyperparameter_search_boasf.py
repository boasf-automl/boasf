#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2020-08-21
import sys
import os

CUR_PATH = os.path.abspath(os.path.dirname(__file__))
ROOT_PATH = os.path.split(CUR_PATH)[0]
sys.path.append(ROOT_PATH)

from src.bandit import generate_orthogonal_bandits
from src.utils.parameter_space import spilt_paramter_space

import argparse
from os import path
from src.utils.eval_utils import eval_performance
from src.utils.data_utils import load_numpy
from src.utils import log_utils
from src.boush import BOUSH
from src.base_algorithms import DEFAULT_CLASSIFICATION_ALGORITHMS
import numpy as np
import warnings
import time
import shutil
import logging
import io
from src.base_algorithms.get_algorithm import get_algorithm_by_key

warnings.filterwarnings("ignore")

"""
Use BOASF to search hyperparameter.
"""

parser = argparse.ArgumentParser(description='Fast AutoSpeech')

parser.add_argument('--dataset_base_path', type=str, default='../sample_data/', help="The base path of datasets.")
parser.add_argument('--test_metric', type=str, default='balanced_accuracy_score',
                    choices=["accuracy_score", "balanced_accuracy_score", "roc_auc_score"],
                    help="The base path of datasets.")
parser.add_argument('--model', type=str, default='LogisticRegression', help="The specified model.")
parser.add_argument('--dataset_idx', type=int, default=475, help="The index of openml dataset.")
parser.add_argument('--max_rounds', type=int, default=3, help='Maximum round.')
parser.add_argument('--time_budget', type=int, default=3600, help='Time budget.')
parser.add_argument('--per_model_time_budget', type=int, default=120, help='Time budget of per model run.')
parser.add_argument('--gucb_c', type=float, default=2.0, help='C for gaussian UCB.')
args = parser.parse_args()

LOG_PATH = path.join(ROOT_PATH, 'logs')


def train(base_path, dataset, bud, model_name='LogisticRegression', test_metric='accuracy_score',
          per_model_time_budget=120, gucb_c=2, max_rounds=3):
    sio = io.StringIO()
    per_model_time_budget = per_model_time_budget
    total_budget = bud
    max_asf_rounds = max_rounds  # totally 16 algorithms，log(2,16)=4
    TASK_NAME = f"openml{dataset}_hyperparameter_search_{model_name}_boasf_{test_metric}_{total_budget}_{max_asf_rounds}"

    log_path = path.join(LOG_PATH, TASK_NAME)
    logger = logging.getLogger('BOASF')
    logger = log_utils.init_logger(logger, log_path, 'DEBUG')

    x_train, x_test, y_train, y_test, cat_cols = load_numpy(base_path, dataset, logger)

    validates_records = []
    test_records = []
    time_cost_records = []

    for i in range(1):
        bandit_inst = {}

        model = get_algorithm_by_key(model_name)
        # donot use the preprocessing space
        model.disable_preprocessing()
        split_spaces = spilt_paramter_space(model, 2)

        generate_orthogonal_bandits(split_spaces, model_name, 0, bandit_inst, {}, gucb_c=gucb_c)
        # donot use the preprocessing space
        for b in bandit_inst.keys():
            bandit_inst[b].disable_preprocessing()

        start_time = time.time()

        boasf = BOUSH(total_budget,
                      bandit_inst,
                      budget_type="time",
                      max_number_of_round=max_asf_rounds,
                      metric_name=test_metric,
                      per_model_time_budget_budget=per_model_time_budget)

        try:
            boasf.run(x_train, y_train)

            key_of_best_bandit = boasf.key_of_best_record_across_all_bandits  # this best is the bandit with highest score record
            params_of_best_bandit = bandit_inst[key_of_best_bandit].get_best_model_parameters()
            logger.info(f"BOUSH best bandit name: {bandit_inst[key_of_best_bandit].model_name}; "
                        f"best bandit model parameters: {params_of_best_bandit}")

            best_clf = bandit_inst[key_of_best_bandit]
            best_clf.set_params(**params_of_best_bandit)
            best_clf.fit(x_train, y_train)
            y_hat = best_clf.predict(x_test)

            test_score = eval_performance(test_metric, y_test, y_hat)
            validate_score = bandit_inst[key_of_best_bandit].get_best_record()
            logger.info(f"BOUSH best bandit test score: {test_score}; validate score: {validate_score}")

            sio.write("[INFO] ==========round {}==========\n".format(i))
            for i in bandit_inst.keys():
                infos = bandit_inst[i].print_statistic()
                logger.info(f"[model_name]: {bandit_inst[i].model_name}; infos: {infos}")
                sio.write(infos)

            validates_records.append(validate_score)
            test_records.append(test_score)
        except Exception as e:
            import traceback
            traceback.print_exc()
            validates_records.append(0.0)
            test_records.append(0.0)
            logger.error(traceback.format_exc())

        time_cost_records.append(time.time() - start_time)

    info = "validates_records: {}".format(validates_records)
    logger.info(info)
    sio.write(info + '\n')

    info = "test_records: {}".format(test_records)
    logger.info(info)
    sio.write(info + '\n')

    info = "time_cost_records: {}".format(time_cost_records)
    logger.info(info)
    # sio.write(info + '\n')

    with open(path.join(log_path, TASK_NAME + '_statistic.log'), 'w') as fd:
        sio.seek(0)
        shutil.copyfileobj(sio, fd)


if __name__ == '__main__':
    base_path = args.dataset_base_path
    idx = args.dataset_idx
    budget = args.time_budget
    test_metric = args.test_metric
    per_model_time_budget = args.per_model_time_budget
    gucb_c = args.gucb_c
    max_rounds = args.max_rounds
    model = args.model

    train(base_path, idx, budget, model_name=model, test_metric=test_metric,
          per_model_time_budget=per_model_time_budget, gucb_c=gucb_c, max_rounds=max_rounds)
