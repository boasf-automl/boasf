#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2020-08-14
import logging

import numpy as np

logger = logging.getLogger('BOASF')


class BOUSH(object):
    """
    reward could not be less than 0
    """

    def __init__(self,
                 total_budget,
                 all_bandit,
                 budget_type="round",
                 max_number_of_round=3,
                 per_model_time_budget_budget=60,
                 metric_name="accuracy_score"):
        self._budget_type = budget_type
        self._total_budget = total_budget
        self._all_bandit = all_bandit
        logger.info(f"Init bandits: {len(all_bandit)}")

        self._keys_of_all_bandit = all_bandit.keys()
        self._all_bandit_ucb_score = {}
        self._reject_list = set()
        self._current_list = set(self._all_bandit.keys())

        self._max_rounds = max_number_of_round

        self._key_of_best_bandit = None
        self._best_bandit = None

        self._key_of_best_record_across_all_bandits = None
        self._bandit_with_best_record = None

        self._per_model_time_budget = per_model_time_budget_budget
        self._metric_name = metric_name

    @property
    def all_bandit(self):
        return self._all_bandit

    @all_bandit.setter
    def all_bandit(self, v):
        if not isinstance(v, dict):
            raise Exception("The parameter bandit should be a list")
        self._all_bandit = v

    def _judge(self, p):
        # p is the probability to go on
        num_of_failed = 0
        if np.random.uniform() > p:
            num_of_failed = 1

        if num_of_failed > 0:
            return True
        else:
            return False

    def _reject(self, cur_round):
        # reject one bandit at least
        tmp_ucb_score = self._get_ucb_score()

        tmp_success_rate = {}
        for i in self._current_list:
            if i in tmp_ucb_score:
                tmp_success_rate[i] = tmp_ucb_score[i]
            else:
                tmp_success_rate[i] = 0

        tmp_success_rate = self._min_max_scale(tmp_success_rate)

        self._current_list.clear()
        while True:
            for i in tmp_success_rate.keys():
                # reject based on probability
                flag = self._judge(tmp_success_rate[i])
                # print("flag", flag)
                if flag:
                    # reject
                    self._reject_list.add(i)
                else:
                    # go on
                    self._current_list.add(i)
            if len(self._current_list) > 0:
                break

        logger.info(f"Current round {cur_round};"
                    # f" tmp_success_rate {tmp_success_rate}; "
                    f"reject {len(self._reject_list)}; keep {len(self._current_list)}")
        # logger.info(f"Current round {cur_round}; tmp_success_rate {tmp_success_rate}; "
        #             f"reject {self._reject_list}; keep {self._current_list}")

    def _allocate_resouce(self, total_resource):
        # tmp_keys = []
        tmp_ucb = {}
        for i in self._current_list:
            # tmp_keys.append(i)
            tmp_ucb[i] = self._all_bandit[i].get_ucb_score()
        pi = self._soft_max(tmp_ucb)
        for i in pi.keys():
            pi[i] = int(pi[i] * total_resource)
        return pi

    def _min_max_scale(self, arr):
        # arr is a dict
        min_val = min(arr.values())
        max_val = max(arr.values())
        res = {}
        if min_val == max_val:
            for i in arr.keys():
                res[i] = 1 / len(arr.keys())
        else:
            for i in arr.keys():
                res[i] = (arr[i] - min_val) / (max_val - min_val)
        return res

    def _soft_max(self, s):
        tmp_arr = list(s.values())
        # https://stackoverflow.com/questions/54880369/implementation-of-softmax-function-returns-nan-for-high-inputs
        # shift values to avoid nan
        max_ = max(tmp_arr)
        tmp_arr = [i - max_ for i in tmp_arr]
        for k in s.keys():
            s[k] = s[k] - max_

        tmp_arr = np.array(tmp_arr)
        tmp_arr = np.nan_to_num(tmp_arr)
        pi = {}
        for i in s.keys():
            pi[i] = np.exp(s[i]) / (np.sum(np.exp(tmp_arr)))
        return pi

    def _get_ucb_score(self):
        self._all_bandit_ucb_score = {}
        for k in self.all_bandit:
            if k in self._reject_list:
                continue
            else:
                self._all_bandit_ucb_score[k] = self._all_bandit[k].get_ucb_score()
        return self._all_bandit_ucb_score

    def get_reource_per_round(self, method="uniform"):
        # exponential: 1 : 2 : 4 : 8 : 16 ...
        resource_per_round = None
        if method == "uniform":
            resource_per_round = [int(self._total_budget / self._max_rounds) for i in
                                  range(self._max_rounds)]

        elif method == "exponentional":
            k = self._max_rounds - 1
            resource_per_round = []
            init = 1
            delta = self._total_budget // (2 ** k - 1)
            for i in range(k):
                resource_per_round.append(init * delta)
                init *= 2

            left = self._total_budget - delta * (2 ** k - 1)
            if left >= 0:
                resource_per_round[-1] += left
                self._max_rounds -= 1
            # resource_per_round.append(self._total_budget - delta*(2**(k) - 1))
        logger.info(f"resource_per_round: {resource_per_round}; "
                    f"sum of resource_per_round: {sum(resource_per_round)}; "
                    f"total_budget: {self._total_budget}; "
                    f"max rounds {self._max_rounds}")
        return resource_per_round

    def get_bandit_key_with_best_record_across_all_bandits(self):
        """
        Get the corresponding bandit with highest performance in all bandits
        """
        res = None
        max_record = None
        for b in self._all_bandit.keys():
            if max_record is None:
                max_record = self._all_bandit[b].get_best_record()
                res = b
            else:
                tmp = self._all_bandit[b].get_best_record()
                if tmp > max_record:
                    max_record = tmp
                    res = b
        return res

    def run(self, X, y, task='classification', random_state=None):
        resource_per_round = self.get_reource_per_round()

        resource_used = 0

        for i in range(self._max_rounds):

            logger.debug("round: {}, self._current_list: {}".format(i, self._current_list))

            tmp_ucb_score = self._get_ucb_score()
            logger.debug("tmp_ucb_score: {}".format(tmp_ucb_score))

            resource_for_every_bandit = self._allocate_resouce(resource_per_round[i])
            logger.debug("resource for every bandit: {}".format(resource_for_every_bandit))

            for k in self._current_list:
                reward, model_parameters, pipeline_parameters = \
                    self._all_bandit[k].compute_boasf(X, y,
                                                      resource_for_every_bandit[k],
                                                      resource_type=self._budget_type,
                                                      metric_name=self._metric_name,
                                                      per_model_time_budget=self._per_model_time_budget,
                                                      task=task)
                if reward is not None:
                    self._all_bandit[k].add_record(reward, model_parameters, pipeline_parameters)

            resource_used += resource_per_round[i]
            if i == self._max_rounds - 1:
                max_val = None
                max_ind = None
                tmp_ucb_score = self._get_ucb_score()
                logger.debug(f"In last round: {self._current_list}; tmp_ucb_score: {tmp_ucb_score}")
                for ii in self._current_list:
                    if max_val is None:
                        max_val = tmp_ucb_score[ii]
                        max_ind = ii
                    else:
                        if max_val < tmp_ucb_score[ii]:
                            max_val = tmp_ucb_score[ii]
                            max_ind = ii
                self._key_of_best_bandit = max_ind

                max_val = None
                max_ind = None
                for ii in self._all_bandit.keys():
                    tmp = self._all_bandit[ii].get_best_record()
                    if max_val is None or max_val < tmp:
                        max_val = tmp
                        max_ind = ii
                self.key_of_best_record_across_all_bandits = max_ind

            self._reject(i)
