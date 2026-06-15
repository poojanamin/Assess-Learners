import numpy as np


class RTLearner:
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

    def author(self):
        return "903961442"

    def add_evidence(self, data_x, data_y):
        if self.verbose:
            print("Building tree for RTLearner")
        self.tree = self.build_tree(data_x, data_y)

    def build_tree(self, data_x, data_y):
        if data_x.shape[0] <= self.leaf_size or len(set(data_y)) == 1:
            return np.array([[-1, np.mean(data_y), np.nan, np.nan]])

        if np.all(data_x[:, 0] == data_x[0, 0]):
            return np.array([[-1, np.mean(data_y), np.nan, np.nan]])

        feature_idx = np.random.randint(data_x.shape[1])
        split_val = np.median(data_x[:, feature_idx])

        if np.all(data_x[:, feature_idx] == split_val):
            return np.array([[-1, np.mean(data_y), np.nan, np.nan]])

        left_indices = data_x[:, feature_idx] <= split_val
        right_indices = data_x[:, feature_idx] > split_val

        # Handle case where split does not divide the data
        if np.all(left_indices) or np.all(right_indices):
            return np.array([[-1, np.mean(data_y), np.nan, np.nan]])

        left_tree = self.build_tree(data_x[left_indices], data_y[left_indices])
        right_tree = self.build_tree(data_x[right_indices], data_y[right_indices])

        root = np.array([[feature_idx, split_val, 1, left_tree.shape[0] + 1]])
        return np.vstack((root, left_tree, right_tree))

    def query(self, points):
        predictions = np.array([self.query_point(point) for point in points])
        return predictions

    def query_point(self, point):
        node = 0
        while self.tree[node][0] != -1:
            feature_index = int(self.tree[node][0])
            split_val = self.tree[node][1]
            if point[feature_index] <= split_val:
                node += int(self.tree[node][2])
            else:
                node += int(self.tree[node][3])
        return self.tree[node][1]
