#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2020-08-13


DEFAULT_CLASSIFICATION_ALGORITHMS = [
    # autosklearn has 16 classification algorithms
    "RandomForestClassifier",
    "AdaboostClassifier",
    "LinearDA",
    "QuadraticDA",
    "ExtraTreesClassifier",
    "GBDTClassifier",
    "BernouliNB",
    "MultinomialNB",
    "GaussianNB",
    "SGDClassifier",
    "SVC",
    "LinearSVC",
    "TreeClassifier",
    "KNeighborsClassifier",
    "PassiveAggressiveClassifier",
    # "GPC",
    "LogisticRegression",
    # "MLPClassifier",
    # "LGBClassifier"
]


DEFAULT_PREPROCESS_ALGORITHMS = [
    "MinMaxScaler",
    "Normalizer",
    "QuantileTransformer",
    "RobustScaler",
    "StandardScaler"
]


DEFAULT_FEATURE_ENGINEERING_ALGORITHMS = {
    "VarianceThreshold"
}
