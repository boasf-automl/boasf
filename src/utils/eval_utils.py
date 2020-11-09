#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2020-08-14
import logging
import numpy as np

from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold

logger = logging.getLogger('BOASF')


def eval_performance(metric_name, y_true, y_pred, average='micro', normalize=True, sample_weight=None):
    if metric_name == 'balanced_accuracy_score':
        return balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    elif metric_name == 'accuracy_score':
        return accuracy_score(y_true, y_pred, normalize=normalize, sample_weight=sample_weight)
    elif metric_name == 'roc_auc_score':
        # TODO check multilabel classfication
        if len(y_pred.shape) == 2 and y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1]
        return roc_auc_score(y_true, y_pred, average=average, sample_weight=sample_weight)
    else:
        raise Exception(f"No such metrics {metric_name}")


def cross_validate_score(model, X, y, cv=5,
                         predict_method="predict", metric_name="accuracy_score", random_state=None):
    if len(y.shape) == 1:
        y_unique = len(set(y))
    elif len(y.shape) == 2 and y.shape[1] == 1:
        y = y.ravel()
        y_unique = len(set(y))
    else:
        y_unique = y.shape[1]

    row_n, col_n= X.shape[0], X.shape[1]

    if predict_method == "predict" and len(y.shape) == 1:
        y_final_val_pred = np.zeros((row_n, ))
    else:
        y_final_val_pred = np.zeros((row_n, y_unique))

    scores = []
    logger.debug("start cross validate")
    if len(y.shape) == 1:
        kf = StratifiedKFold(n_splits=cv, random_state=random_state)
    else:
        kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    logger.debug("split data")

    y_target = [None] * cv
    y_target_pred = [None] * cv

    for i, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[test_idx]
        y_val = y[test_idx]

        logger.debug("start fit model")
        clf = model
        clf.fit(X_train, y_train)
        logger.debug("end fit model")

        if predict_method == "predict":
            y_val_pred = clf.predict(X_val)
        elif predict_method == "predict_proba":
            y_val_pred = clf.predict_proba(X_val)
        else:
            raise Exception("invalid predict method")
        logger.debug("y_val_pred.shape: {}; predict_method {}".format(y_val_pred.shape, predict_method))

        y_val_pred = np.nan_to_num(y_val_pred)
        y_target[i] = y[test_idx]
        y_target_pred[i] = y_val_pred
        if len(y.shape) == 1:
            y_final_val_pred[test_idx] = y_val_pred
        else:
            for j in range(len(test_idx)):
                y_final_val_pred[test_idx[j]] = y_val_pred[i]

    y_target = np.concatenate([y_target[i] for i in range(cv) if y_target[i] is not None])
    y_target_pred = np.concatenate([y_target_pred[i] for i in range(cv) if y_target_pred[i] is not None])

    score = eval_performance(metric_name, y_target, y_target_pred)
    scores.append(score)

    return np.mean(scores), y_final_val_pred
