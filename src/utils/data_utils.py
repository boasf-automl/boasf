#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2020-08-14
import os

import numpy as np
from sklearn.model_selection import train_test_split


def load_numpy(base_path, id, logger):
    X = np.load(os.path.join(base_path, str(id) + '_X.npy'), allow_pickle=True)
    y = np.load(os.path.join(base_path, str(id) + '_y.npy'), allow_pickle=True)
    cat = np.load(os.path.join(base_path, str(id) + '_cat.npy'), allow_pickle=True).tolist()

    if y.dtype == 'float64':
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        le.fit(y)
        y = le.transform(y)

    filtered_cat = []
    if cat is not None:
        for c in cat:
            if np.unique(X[:, c]).shape[0] < 20:
                filtered_cat.append(c)

    if filtered_cat == []:
        filtered_cat = None

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    unique_values, occur_count = np.unique(y_train, return_counts=True)
    max_in_columns = np.amax(x_train, axis=0)

    logger.info(f"Dataset path {base_path}; Dataset id {id}; "
                f"x_train {x_train.shape}; y_train {y_train.shape}; "
                f"cat_cols {filtered_cat}; "
                f"Unique values {unique_values}; "
                f"Occurrence count {occur_count}; "
                f"max in columns: {max_in_columns}")

    return x_train, x_test, y_train, y_test, filtered_cat
