#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2020-08-13
# -*- coding: utf-8 -*-
import sys
import os

CUR_PATH = os.path.abspath(os.path.dirname(__file__))
ROOT_PATH = os.path.split(CUR_PATH)[0]
sys.path.append(ROOT_PATH)

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
Use BOASF to select model.
"""

parser = argparse.ArgumentParser(description='BOASF')

parser.add_argument('--dataset_base_path', type=str, default='../sample_data/', help="The base path of datasets.")
parser.add_argument('--test_metric', type=str, default='balanced_accuracy_score',
                    choices=["accuracy_score", "balanced_accuracy_score", "roc_auc_score"],
                    help="The base path of datasets.")
parser.add_argument('--dataset_idx', type=int, default=475, help="The index of openml dataset.")
parser.add_argument('--time_budget', type=int, default=3600, help='Time budget.')
parser.add_argument('--max_rounds', type=int, default=3, help='Maximum round.')
parser.add_argument('--per_model_time_budget', type=int, default=120, help='Time budget of per model run.')
parser.add_argument('--gucb_c', type=float, default=2.0, help='C for gaussian UCB.')
args = parser.parse_args()

LOG_PATH = path.join(ROOT_PATH, 'logs')


def train(base_path, dataset, bud, test_metric='accuracy_score', per_model_time_budget=120, gucb_c=2, max_rounds=3):
    sio = io.StringIO()
    total_budget = bud
    max_asf_rounds = max_rounds  # totally 16 algorithms，log(2,16)=4
    TASK_NAME = f"openml{dataset}_model_selection_boasf_{test_metric}_{total_budget}_{max_asf_rounds}"

    log_path = path.join(LOG_PATH, TASK_NAME)
    logger = logging.getLogger('BOASF')
    logger = log_utils.init_logger(logger, log_path, 'DEBUG')

    x_train, x_test, y_train, y_test, cat_cols = load_numpy(base_path, dataset, logger)

    validates_records = []
    test_records = []
    time_cost_records = []

    for i in range(1):
        bandit_inst = {}
        for idx, clf in enumerate(DEFAULT_CLASSIFICATION_ALGORITHMS):
            tmp = get_algorithm_by_key(clf)
            tmp.gucb_c = gucb_c
            bandit_inst[idx] = tmp

        start_time = time.time()

        boasf = BOUSH(total_budget,
                      bandit_inst,
                      budget_type="time",
                      max_number_of_round=max_asf_rounds,
                      per_model_time_budget_budget=per_model_time_budget,
                      metric_name=test_metric)

        try:
            boasf.run(x_train, y_train)

            key_of_best_bandit = boasf.get_bandit_key_with_best_record_across_all_bandits()
            model_params_of_best_bandit = bandit_inst[key_of_best_bandit].get_best_model_parameters()
            pipeline_params_of_best_bandit = bandit_inst[
                key_of_best_bandit].get_best_pipeline_parameters()  # include balancing, rescaling, variance_threshold
            logger.info(f"BOUSH best bandit name: {bandit_inst[key_of_best_bandit].model_name}; "
                        f"best bandit model parameters: {model_params_of_best_bandit}; "
                        f"best bandit pipeline parameters: {pipeline_params_of_best_bandit}")

            params = {**model_params_of_best_bandit, **pipeline_params_of_best_bandit}
            best_clf = bandit_inst[key_of_best_bandit]
            best_clf.set_params(**params)

            best_clf.fit(x_train, y_train)
            y_hat = best_clf.predict(x_test)

            test_score = eval_performance(test_metric, y_test, y_hat)
            validate_score = bandit_inst[key_of_best_bandit].get_best_record()
            logger.info(f"BOUSH best bandit test score: {test_score}; validate score: {validate_score}")

            sio.write("==========round {}==========\n".format(i))
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

    train(base_path, idx, budget, test_metric=test_metric, per_model_time_budget=per_model_time_budget, gucb_c=gucb_c,
          max_rounds=max_rounds)
