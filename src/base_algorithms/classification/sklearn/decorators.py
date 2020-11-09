# -*- coding: utf-8 -*-
# @Time    : 2019/11/13 3:16 下午
# @Author  : fang xin
# @FileName: decorators.py
# @Software: PyCharm
import copy
from functools import wraps

import numpy as np
from ConfigSpace.hyperparameters import CategoricalHyperparameter


has_sample_weights_fit = [
    "AdaboostClassifier",
    "GBDTClassifier",
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "SGDClassifier",
    # "PassiveAggressiveClassifier",
    "XGBClassifier"
]

has_class_weight = [
    "TreeClassifier",
    "LinearSVC",
    "SVC",
    "NuSVC"
]


def merge_balancing_into_smac_space(func):
    @wraps(func)
    def inner(self, *args, **kwargs):
        if self.smac_config_space is None:
            raw_space = func(self, *args, **kwargs)
            raw_space.add_hyperparameter(
                CategoricalHyperparameter(name='balancing', choices=[True, False], default_value=True))
            self.smac_config_space = raw_space

        return self.smac_config_space

    return inner


def merge_preprocessing_spaces(func):
    @wraps(func)
    def inner(self, *args, **kwargs):
        if self.parameter_space is None:
            self.set_configuration_space()
        if not self._has_mergered_param_sapce:
            self.parameter_space.merge(self._preprocessing_spaces)
            self._has_mergered_param_sapce = True
        return func(self, *args, **kwargs)

    return inner


def additional_preprocessing_fit(func):
    @wraps(func)
    def inner(self, X, Y, *args, **kwargs):
        self._preprocessing_models = []  # first: variance_threshold, second: rescaling
        from src.base_algorithms.get_algorithm import get_algorithm_by_key
        if self._variance_threshold:
            vt = get_algorithm_by_key("VarianceThreshold")
            X = vt.fit_transform(X)
            self._preprocessing_models.append(vt)

        if self._rescaling != "None":
            rs = get_algorithm_by_key(self._rescaling)
            X = rs.fit_transform(X)
            self._preprocessing_models.append(rs)

        if self._balancing:
            # reference: https://github.com/automl/auto-sklearn/blob/master/autosklearn/pipeline/components/data_preprocessing/balancing/balancing.py
            if len(Y.shape) > 1:
                offsets = [2 ** i for i in range(Y.shape[1])]
                Y_ = np.sum(Y * offsets, axis=1)
            else:
                Y_ = Y

            unique, counts = np.unique(Y_, return_counts=True)
            # This will result in an average weight of 1!
            cw = 1 / (counts / np.sum(counts)) / 2

            if len(Y.shape) == 2:
                cw /= Y.shape[1]

            sample_weights = np.ones(Y_.shape)

            for i, ue in enumerate(unique):
                mask = Y_ == ue
                sample_weights[mask] *= cw[i]

            if self.model_name in has_sample_weights_fit:
                return func(self, X, Y, sample_weight=sample_weights)

            if self.model_name in has_class_weight:
                self.model.class_weight = 'balanced'
                return func(self, X, Y, *args, **kwargs)

        return func(self, X, Y, *args, **kwargs)

    return inner


def additional_preprocessing_predict(func):
    @wraps(func)
    def inner(self, X, *args, **kwargs):
        for pre_mod in self._preprocessing_models:
            X = pre_mod.transform(X)

        return func(self, X, *args, **kwargs)

    return inner


def additional_preprocessing_set_params(func):
    @wraps(func)
    def inner(self, **kwargs):
        # TODO whether keep
        self.smac_params = copy.deepcopy(kwargs)
        for prepocessing_param in self._preprocessing_spaces:
            # print(prepocessing_param.get_name()) # final_algorithm__RandomForestClassifier__balancing
            if prepocessing_param.get_name().endswith('balancing') and 'balancing' in kwargs:
                self.balancing = kwargs['balancing']
                kwargs.pop('balancing', None)
            elif prepocessing_param.get_name().endswith('rescaling') and 'rescaling' in kwargs:
                self.rescaling = kwargs['rescaling']
                kwargs.pop('rescaling', None)
            elif prepocessing_param.get_name().endswith('variance_threshold') and 'variance_threshold' in kwargs:
                self.variance_threshold = kwargs['variance_threshold']
                kwargs.pop('variance_threshold', None)

        # delete model_name attribute because sklearn model don't have this attribute
        kwargs_tmp = copy.deepcopy(kwargs)
        if 'model_name' in kwargs_tmp:
            del kwargs_tmp['model_name']

        return func(self, **kwargs_tmp)

    return inner


def additional_preprocessing_get_params(func):
    @wraps(func)
    def inner(self, **kwargs):
        raw_param_dict = func(self, **kwargs)
        for prepocessing_param in self._preprocessing_spaces:
            # print(prepocessing_param.get_name()) # final_algorithm__RandomForestClassifier__balancing
            if prepocessing_param.get_name().endswith('balancing') and 'balancing' not in raw_param_dict:
                # assert 'balancing' not in raw_param_dict, "BaseAlgorithm already has a param named 'balancing'"
                raw_param_dict['balancing'] = self._balancing
            elif prepocessing_param.get_name().endswith('rescaling') and 'rescaling' not in raw_param_dict:
                # assert 'rescaling' not in raw_param_dict, "BaseAlgorithm already has a param named 'rescaling'"
                raw_param_dict['rescaling'] = self._rescaling
            elif prepocessing_param.get_name().endswith(
                    'variance_threshold') and 'variance_threshold' not in raw_param_dict:
                # assert 'variance_threshold' not in raw_param_dict, "BaseAlgorithm already has a param named 'variance_threshold'"
                raw_param_dict['variance_threshold'] = self._variance_threshold

        # print(raw_param_dict)
        return raw_param_dict

    return inner
