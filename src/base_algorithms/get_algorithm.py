#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2020-08-13
import logging

from sklearn.feature_selection import variance_threshold

from src.base_algorithms import DEFAULT_CLASSIFICATION_ALGORITHMS, DEFAULT_PREPROCESS_ALGORITHMS, \
    DEFAULT_FEATURE_ENGINEERING_ALGORITHMS
from src.base_algorithms.classification.sklearn import discriminant_analysis, naive_bayes, svm, \
    tree, lr
from src.base_algorithms.classification.sklearn import passive_aggressive, gbdt, adaboost, sgd, gaussian_process, knn, \
    mlp
from src.base_algorithms.preprocess.sklearn import min_max_scaler, normalizer, quantile_transformer, standard_scaler, \
    robust_scaler, max_abs_scaler

logger = logging.getLogger('BOASF')


def get_algorithm_by_key(key, random_state=None, row_num=0, col_num=0):
    if key in DEFAULT_CLASSIFICATION_ALGORITHMS:
        return __get_classification_algorithm(key, random_state=random_state)
    elif key in DEFAULT_PREPROCESS_ALGORITHMS:
        return __get_preprocess_algorithm(key)
    elif key in DEFAULT_FEATURE_ENGINEERING_ALGORITHMS:
        return __get_feature_engineer_algorithm(key)
    else:
        logger.error(f"No such algorithm {key}")
        raise Exception(f"No such algorithm {key}")


def __get_classification_algorithm(key, random_state=None):
    if key == "SGDClassifier":
        return sgd.SGDClassifier(random_state=random_state)
    elif key == "GBDTClassifier":
        return gbdt.GBDT(random_state=random_state)
    elif key == "LogisticRegression":
        return lr.LogisticRegression(random_state=random_state)
    elif key == "TreeClassifier":
        return tree.DecisionTreeClassifier(random_state=random_state)
    elif key == "RandomForestClassifier":
        return tree.RandomForestClassifier(random_state=random_state)
    elif key == "SVC":
        return svm.SVC(random_state=random_state)
    elif key == "AdaboostClassifier":
        return adaboost.AdaBoostClassifier(random_state=random_state)
    elif key == "LinearDA":
        return discriminant_analysis.LinearDiscriminantAnalysis()
    elif key == "QuadraticDA":
        return discriminant_analysis.QuadraticDiscriminantAnalysis()
    elif key == "ExtraTreesClassifier":
        return tree.ExtraTreesClassifier(random_state=random_state)
    elif key == "BernouliNB":
        return naive_bayes.BernouliNB()
    elif key == "MultinomialNB":
        return naive_bayes.MultinomialNB()
    elif key == "GaussianNB":
        return naive_bayes.GaussianNB()
    elif key == "GPC":
        return gaussian_process.GPC(random_state=random_state)
    elif key == "MLPClassifier":
        return mlp.MLPClassifier(random_state=random_state)
    elif key == "KNeighborsClassifier":
        return knn.KNeighborsClassifier()
    elif key == 'PassiveAggressiveClassifier':
        return passive_aggressive.PassiveAggressiveClassifier(
            random_state=random_state)
    elif key == 'LinearSVC':
        return svm.LinearSVC(random_state=random_state)
    else:
        logger.error(f"No such classification algorithm {key}")
        raise Exception(f"No such algorithm {key}")


def __get_preprocess_algorithm(key):
    if key == "MaxAbsScaler":
        return max_abs_scaler.MaxAbsScaler()
    elif key == "MinMaxScaler":
        return min_max_scaler.MinMaxScaler()
    elif key == "Normalizer":
        return normalizer.Normalizer()
    elif key == "QuantileTransformer":
        return quantile_transformer.QuantileTransformer()
    elif key == "RobustScaler":
        return robust_scaler.RobustScaler()
    elif key == "StandardScaler":
        return standard_scaler.StandardScaler()
    else:
        logger.error(f"No such preprocess algorithm {key}")
        raise Exception(f"No such algorithm {key}")


def __get_feature_engineer_algorithm(key, random_state=None):
    if key == "VarianceThreshold":
        return variance_threshold.VarianceThreshold()
    else:
        logger.error(f"No such feature engineering algorithm {key}")
        raise Exception(f"No such algorithm {key}")
