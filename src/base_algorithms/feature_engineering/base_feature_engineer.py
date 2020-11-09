from src.utils.parameter_space import convert_param_space_to_config_space


def warning_no_model(func):
    def decorator(self, *args, **kwargs):
        if self.model is None:
            raise Exception
            # pass
        return func(self, *args, **kwargs)

    return decorator


class BaseFeatureEngineer(object):

    def __init__(self):
        self.model = None
        self._model_name = None
        self.parameter_space = None
        self.col_num = 1
        self.row_num = 1

        self.smac_config_space = None
        self.smac_params = None

    def get_configuration_space(self):
        raise NotImplementedError

    def set_configuration_space(self):
        raise NotImplementedError

    @warning_no_model
    def fit(self, X, y=None):
        return self.model.fit(X, y)

    @warning_no_model
    def transform(self, X):
        return self.model.transform(X)

    @warning_no_model
    def fit_transform(self, X, y=None):
        return self.model.fit_transform(X, y)

    # @warning_no_model
    # def get_params(self, deep=True):
    #     def deco():
    #         return self.model.get_params(deep=deep)
    #
    #     return deco

    # @warning_no_model
    # def get_params(self, deep=True):
    #     return self.model.get_params(deep=deep)

    def get_params(self, deep=True):
        if self.model is None:
            raise Exception(
                "[BaseFeatureEngineer] model is None, no get_params() method!")
        param_dict = self.model.get_params(deep=deep)
        new_param_dict = {}

        if self.smac_params is not None:
            # use smac_configuration_space
            # space_names = list(self.smac_params.keys())
            return self.smac_params
        else:
            space_names = self.get_configuration_space().get_space_names()

        for key in param_dict.keys():
            yes = False
            for space_name in space_names:
                if key in space_name:
                    yes = True
                    break
            if yes:
                new_param_dict[key] = param_dict[key]
        return new_param_dict

    @warning_no_model
    def set_params(self, **kwargs):
        import copy
        self.smac_params = copy.deepcopy(kwargs)
        return self.model.set_params(**kwargs)

    def print_model(self):
        print(self.model_name)

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, v):
        self._model_name = v

    def get_model(self):
        return self.model

    def update_parameter_space(self, name):
        max_fea = min(self.col_num, self.row_num)
        self.print_model()
        for n in self.parameter_space.get_space():
            if n.get_name() == name:
                n.max_val = max_fea
                n.default = max_fea

    def __repr__(self):
        if self.model_name is not None:
            return str(self.model_name)
        return str(self.__class__.__name__)

    def get_smac_config_space(self):
        if self.smac_config_space is None:
            cs = convert_param_space_to_config_space(self.get_configuration_space())

            self.smac_config_space = cs

        return self.smac_config_space