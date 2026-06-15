import numpy as np


class DTLearner:
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

    def author(self):
        return "903961442"  # Your GT username here

    def add_evidence(self, data_x, data_y):
        if self.verbose:
            print("Building tree")
        self.tree = self.build_tree(data_x, data_y)

    def build_tree(self, data_x, data_y):
        # Check if leaf should be created due to leaf size or uniform y-values
        if data_x.shape[0] <= self.leaf_size or len(np.unique(data_y)) == 1:
            return np.array([[-1, np.mean(data_y), np.nan, np.nan]])

        # Attempt to find the best feature for splitting
        best_feature = np.argmax(np.abs(np.corrcoef(data_x.T, data_y)[-1][:-1]))
        split_val = np.median(data_x[:, best_feature])

        # If all x-values for the best feature are the same, create a leaf node
        if np.all(data_x[:, best_feature] == data_x[0, best_feature]):
            return np.array([[-1, np.mean(data_y), np.nan, np.nan]])

        # Split the data on the best feature
        left_mask = data_x[:, best_feature] <= split_val
        right_mask = data_x[:, best_feature] > split_val

        # Check if the split actually partitions the data
        if not np.any(left_mask) or not np.any(right_mask):
            # No effective split possible, create a leaf node
            return np.array([[-1, np.mean(data_y), np.nan, np.nan]])

        # Recursively build the left and right subtrees
        left_tree = self.build_tree(data_x[left_mask], data_y[left_mask])
        right_tree = self.build_tree(data_x[right_mask], data_y[right_mask])

        # Create the root node and combine the subtrees
        root = np.array([[best_feature, split_val, 1, left_tree.shape[0] + 1]])
        return np.vstack((root, left_tree, right_tree))

    def query(self, points):
        predictions = np.apply_along_axis(self.query_point, 1, points)
        return predictions

    def query_point(self, point):
        node = 0
        while True:
            node_val = self.tree[node]
            if int(node_val[0]) == -1:
                return node_val[1]
            feature = int(node_val[0])
            split_val = node_val[1]
            if point[feature] <= split_val:
                node += int(node_val[2])
            else:
                node += int(node_val[3])
