import numpy as np

class BagLearner:
    def __init__(self, learner, kwargs={}, bags=20, boost=False, verbose=False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.learners = [self.learner(**self.kwargs) for _ in range(self.bags)]

    def add_evidence(self, data_x, data_y):
        n = data_x.shape[0]
        for learner in self.learners:
            indices = np.random.choice(n, size=n, replace=True)
            bootstrap_x = data_x[indices]
            bootstrap_y = data_y[indices]
            learner.add_evidence(bootstrap_x, bootstrap_y)

    def query(self, points):
        predictions = np.array([learner.query(points) for learner in self.learners])
        return np.mean(predictions, axis=0)

    def author(self):
        return "903961442"

