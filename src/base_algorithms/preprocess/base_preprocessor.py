from src.utils.parameter_space import convert_param_space_to_config_space


class BasePreprocessor(object):

    def __init__(self):
        self.model = None
        self._model_name = None
        self.parameter_space = None
        self.smac_config_space = None
        self.smac_params = None

    # def get_params(self, deep=True):
    #
    #     if self.model is None:
    #         raise Exception
    #     return self.model.get_params(deep=deep)
    def get_params(self, deep=True):
        if self.model is None:
            raise Exception(
                "[BasePreprocessor] model is None, no get_params() method!")
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

    def set_params(self, **kwargs):

        if self.model is None:
            raise Exception
        import copy
        self.smac_params = copy.deepcopy(kwargs)
        return self.model.set_params(**kwargs)

    def get_configuration_space(self):
        raise NotImplementedError

    def set_configuration_space(self):
        raise NotImplementedError

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

    def __repr__(self):
        if self.model_name is not None:
            return str(self.model_name)
        return str(self.__class__.__name__)

    def get_smac_config_space(self):
        if self.smac_config_space is None:
            cs = convert_param_space_to_config_space(self.get_configuration_space())

            self.smac_config_space = cs

        return self.smac_config_space