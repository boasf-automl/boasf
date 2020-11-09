import numpy as np


class DataMatrix(object):
    """
    Parameters
    ----------
    data: numpy array / scipy.sparse
        Data source of DataMatrix.
    label: list or numpy 1-D array, optional
        Label of the training data.
    missing: float, optional
        Value in the data which needs to be present as a missing value. If None, defaults to np.nan.
    silent: boolean, optional
        Whether print messages during construction
    feature_names: list, optional
        Set names for features.
    feature_types: list, currently only support 'Numeric' or 'Nominal', optional
        Set types for features. If None, defaults all to be 'Numeric'
    categorical_columns: list of int
        List of index of the columns of data, start from 0.
    """

    def __init__(self, data, label=None, missing=None, silent=False, feature_names=None, feature_types=None,
                 categorical_columns=None):
        self.data = data
        self.label = label
        self._silent = silent
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.categorical_columns = categorical_columns

        if missing is not None:
            self.data[np.isnan(self.data)] = missing

        if self.feature_names is None:
            self.feature_names = [str(i) for i in range(self.data.shape[1])]
        else:
            assert len(self.feature_names) == self.data.shape[1], "feature_names length must equal to data.shape[1]"

        if self.feature_types is None:
            self.feature_types = ['Numeric' for i in range(self.data.shape[1])]
        else:
            assert len(self.feature_types) == self.data.shape[1], "feature_types length must equal to data.shape[1]"

        if self.categorical_columns is not None:
            assert len(self.categorical_columns) <= self.data.shape[
                1], "categorical_columns length must less or equal to data.shape[1]"
            for col in self.categorical_columns:
                self.feature_types[col] = 'Nominal'

        self._nominal_columns = [i for i in range(len(self.feature_types)) if self.feature_types[i] == 'Nominal']
        self._numeric_columns = [i for i in range(len(self.feature_types)) if self.feature_types[i] == 'Numeric']

    def __str__(self):
        return 'DataMatrix data shape: {}, ' \
               'feature_names: {}, ' \
               'feature_types: {}, ' \
               'nominal_columns: {}, ' \
               'numeric_columns: {}'.format(self.data.shape, self.feature_names, self.feature_types,
                                            self._nominal_columns, self._numeric_columns)

    def get_numeric_data(self):
        return self.data[:, self._numeric_columns]

    def get_nominal_data(self):
        return self.data[:, self._nominal_columns]

    def get_all_data(self):
        all_cols = sorted(self._numeric_columns + self._nominal_columns)
        return self.data[:, all_cols]

    def set_numeric_columns(self, num_col):
        self._numeric_columns = num_col

    def get_numeric_columns(self):
        return self._numeric_columns

    def set_nominal_columns(self, nom_col):
        self._nominal_columns = nom_col

    def get_nominal_columns(self):
        return self._nominal_columns


if __name__ == '__main__':
    data = np.asarray([[1, 2, np.nan], [4, 5, 6]])

    dm = DataMatrix(data, missing=0.0, categorical_columns=None, feature_names=['3', '2', '1'])

    print(dm)
    print(dm.data)
    print(dm.get_numeric_data().size)
    print(dm.get_nominal_data().shape)
    print(np.concatenate((dm.get_numeric_data(), dm.get_nominal_data()), axis=1))
