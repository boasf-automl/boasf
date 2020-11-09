from src.base_algorithms.classification.base_algorithm import BaseAlgorithm
from sklearn.dummy import DummyClassifier as sk_DummyClassifier


class DummyClassifer(BaseAlgorithm):

    def __init__(self,
                 random_state=None):
        super(DummyClassifer, self).__init__()
        self.random_state = random_state

        self._model_name = "DummyClassifer"
        self.model = sk_DummyClassifier(random_state=self.random_state)

    def fit(self, X, y, sample_weight=None):

        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    # def predict(self, X):
    #     if self.model is None:
    #         raise Exception
    #     return self.model.predict(X)
    #
    # def predict_proba(self, X):
    #     if self.model is None:
    #         raise Exception
    #     return self.model.predict_proba(X)


if __name__ == "__main__":
    pass
