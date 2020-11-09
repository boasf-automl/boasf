import logging
from functools import wraps, partial

from scipy import sparse

from src.utils.data_matrix import DataMatrix

logger = logging.getLogger('BOASF')


def data_matrix_fit(func, argument):
    @wraps(func)
    def inner(self, *args, **kwargs):
        if self.model is None:
            raise Exception('The base model is None!')
        else:
            if isinstance(args[0], DataMatrix):
                logger.debug("fit decorator arg is %s" % str(argument))
                if argument == 'Numeric':
                    selected_data = args[0].get_numeric_data()
                elif argument == 'Nominal':
                    selected_data = args[0].get_nominal_data()
                elif argument == 'All':
                    selected_data = args[0].data
                else:
                    raise Exception('Unsupported data type!')

                if selected_data.size != 0:
                    return func(self, selected_data, *args[1:], **kwargs)
                else:
                    return self
            else:
                return func(self, *args, **kwargs)
    return inner


numeric_data_matrix_fit = partial(data_matrix_fit, argument='Numeric')
nominal_data_matrix_fit = partial(data_matrix_fit, argument='Nominal')
all_type_data_matrix_fit = partial(data_matrix_fit, argument='All')


def data_matrix_transform(func, argument):
    def inner(self, *args, **kwargs):
        if self.model is None:
            raise Exception('The base model is None!')
        else:
            if isinstance(args[0], DataMatrix):
                logger.debug("transform decorator arg is %s" % str(argument))
                if argument == 'Numeric':
                    selected_data = args[0].get_numeric_data()
                    rest_data = args[0].get_nominal_data()
                elif argument == 'Nominal':
                    selected_data = args[0].get_nominal_data()
                    rest_data = args[0].get_numeric_data()
                elif argument == 'All':
                    selected_data = args[0].data
                    rest_data = None
                else:
                    raise Exception('Unsupported data type!')

                # numeric_data = args[0].get_numeric_data()

                if selected_data.size != 0:
                    transformed_selected_data = func(
                        self, selected_data, *args[1:], **kwargs)
                    if sparse.issparse(transformed_selected_data):
                        transformed_selected_data = transformed_selected_data.toarray()
                else:
                    # size = 0, empty
                    transformed_selected_data = selected_data
                """
                # nominal_data = args[0].get_nominal_data()
                if rest_data is not None:

                    transformed_data = np.concatenate((transformed_selected_data, rest_data), axis=1)
                else:
                    transformed_data = transformed_selected_data

                if argument == 'Numeric':
                    # rest data are nominal
                    cat_cols = [i+transformed_selected_data.shape[1] for i in range(rest_data.shape[1])]
                elif argument == 'Nominal':
                    # reserve selected data
                    # concat the raw selected data and transformed data
                    # transformed_data = np.concatenate((selected_data, transformed_data), axis=1)
                    cat_cols = [i for i in range(selected_data.shape[1])]
                else:
                    cat_cols = None
                transformed_data_matrix = DataMatrix(data=transformed_data, categorical_columns=cat_cols)
                return transformed_data_matrix
                """
                return DataMatrix(data=transformed_selected_data,
                                  categorical_columns=None)

            else:
                return func(self, *args, **kwargs)
    return inner


numeric_data_matrix_transform = partial(
    data_matrix_transform, argument='Numeric')
nominal_data_matrix_transform = partial(
    data_matrix_transform, argument='Nominal')
all_type_data_matrix_transform = partial(data_matrix_transform, argument='All')
