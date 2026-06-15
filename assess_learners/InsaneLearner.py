import numpy as np
from BagLearner import BagLearner
from LinRegLearner import LinRegLearner
class InsaneLearner:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.learners = [BagLearner(learner=LinRegLearner, kwargs={}, bags=20, verbose=verbose) for _ in range(20)]

    def add_evidence(self, data_x, data_y):
        for learner in self.learners:
            learner.add_evidence(data_x, data_y)

    def query(self, points):
        predictions = np.array([learner.query(points) for learner in self.learners])
        return np.mean(predictions, axis=0)

    def author(self):
        return "903961442"
