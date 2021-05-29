import numpy as np
from scipy.stats import mode
from collections import Counter
from sklearn.metrics import accuracy_score


class RFClassifier:

    def __init__(self, max_features=None,
                 n_trees=2,
                 min_leaf=1,
                 max_depth=np.inf,
                 inf_value_type="Gini",
                 oob_vote: float = None,
                 random_state=None):

        self.n_trees = n_trees
        self.min_leaf = min_leaf
        self.inf_value_type = inf_value_type
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.oob_vote = oob_vote

        self.prediction = None
        self.y_proba = None
        self.proba_data = None
        self.X = None
        self.y = None
        self.forest_pred = []
        self.forest_proba = []
        self.forest_list_to_predict = []
        self.oob_idx = []
        self.oob_scores_list = []

    @staticmethod
    def get_label_num(y_list):
        num = Counter(y_list)
        return dict(num)

    @staticmethod
    def get_bootstrap_idx(n_samples, rdm):
        rng = np.random.RandomState(rdm)
        indexes = rng.randint(0, n_samples - 1, size=n_samples)
        return indexes

    def get_subsample(self, n_features, rdm):
        rng = np.random.RandomState(rdm)
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))
        subsample = rng.choice(
            n_features, self.max_features, replace=False)

        return subsample

    def get_informative_value(self, y_list):
        labels_dict = self.get_label_num(y_list)

        if self.inf_value_type == "Gini":
            impurity = 1
            for label in labels_dict:
                p_label = labels_dict[label] / len(y_list)
                impurity -= p_label ** 2
            return impurity

        if self.inf_value_type == "Shannon":
            entropy = 0
            for label in labels_dict:
                p_label = labels_dict[label] / len(y_list)
                entropy -= p_label * (0 if (p_label == 0)
                                      else np.log2(p_label))
            return entropy

    def merit_functional(self, true_labels, false_labels, current_informative_value):

        p = (true_labels.shape[0]) / \
            ((true_labels.shape[0]) + (false_labels.shape[0]))
        quality = current_informative_value - p * self.get_informative_value(true_labels) - \
                  (1 - p) * self.get_informative_value(false_labels)
        return quality

    def find_best_split(self, X, y, subsample):

        current_informative_value = self.get_informative_value(y)
        best_quality = 0
        best_t = None
        best_index = None
        n_features = X.shape[1]

        for index in subsample:
            t_values = set(X[:, index])

            for t in t_values:
                true_data, false_data, true_labels, false_labels = self.data_split(
                    X, y, index, t)
                if len(true_data) < self.min_leaf or len(false_data) < self.min_leaf:
                    continue

                current_quality = self.merit_functional(
                    true_labels, false_labels, current_informative_value)

                if current_quality > best_quality:
                    best_quality, best_t, best_index = current_quality, t, index

        return best_quality, best_t, best_index

    @staticmethod
    def data_split(X, y, feature_idx, t):
        left = (X[:, feature_idx] <= t)
        true_data = X[left]
        false_data = X[~left]
        true_labels = y[left]
        false_labels = y[~left]
        return true_data, false_data, true_labels, false_labels

    def tree_fit(self, X, y, subsample):
        leaf_list = []
        prediction_mask = {}
        node_dict = {}
        level = 0
        data_id = 0
        data_list = [[X, y, None, None, data_id]]
        next_level_data = []
        fl = 0

        while not fl:
            if level > 0:
                data_list.clear()
                data_list = next_level_data.copy()
                next_level_data.clear()

            for data in data_list:

                best_quality, best_t, best_index = self.find_best_split(
                    data[0], data[1], subsample)
                if best_quality and (level <= self.max_depth):
                    true_data, false_data, true_labels, false_labels = self.data_split(
                        data[0], data[1], best_index, best_t)
                    node_dict[data[4]] = [
                        data[2], data[3], best_index, best_t]
                    data_id += 1
                    next_level_data.append(
                        [true_data, true_labels, True, data[4], data_id])
                    data_id += 1
                    next_level_data.append(
                        [false_data, false_labels, False, data[4], data_id])
                else:
                    leaf = self.get_label_num(data[1])
                    prediction = max(leaf, key=leaf.get)
                    proba = leaf[prediction] / len(data[0])
                    if prediction == 0:
                        proba = 1 - proba
                    leaf_list.append(
                        [data[2], data[3], proba, prediction])

                level += 1

            if not next_level_data:
                fl = 1

        for leaf in leaf_list:
            mask = [(leaf[0], leaf[1])]
            next_step = leaf[1]

            while True:

                if next_step is not None and next_step != 0:
                    mask.append(
                        (node_dict[next_step][0], node_dict[next_step][1]))
                    next_step = node_dict[next_step][1]
                else:
                    mask.reverse()
                    prediction_mask[tuple(mask)] = leaf[2], leaf[3]
                    break
        self.forest_list_to_predict.append([prediction_mask, node_dict])

    def fit(self, X, y):

        self.X = X
        self.y = y

        n_samples = X.shape[0]
        n_features = X.shape[1]
        rdm = None
        if self.random_state:
            rng = np.random.RandomState(self.random_state)
            rdm = rng.randint(1, 1000, self.n_trees)

        for tree in range(self.n_trees):
            r_state = rdm[tree] if self.random_state else self.random_state
            idx = self.get_bootstrap_idx(n_samples, r_state)
            self.oob_idx.append(list(set(range(n_samples)).difference(idx)))
            subsample = self.get_subsample(n_features, r_state)
            self.tree_fit(self.X[idx, :], self.y[idx], subsample)

    def predict(self, X_test):

        forest_proba = []
        forest_pred = []
        answer_mask_array = None
        y_pred_proba = []

        for n, tree in enumerate(self.forest_list_to_predict):
            y_pred_proba_list = [None for _ in range(len(X_test))]

            if self.oob_vote:  # oob
                X_oob = self.X[self.oob_idx[n], :]
                y_pred_oob = [None for i in range(len(X_oob))]
                for mask in tree[0]:

                    answer_mask = []
                    predict_mask = [mask[i][0] for i in range(len(mask))]

                    for question in mask:
                        idx = tree[1][question[1]][2]
                        t = tree[1][question[1]][3]
                        y_pred_proba = [tree[0][mask]]
                        answer_mask.append(X_oob[:, idx] <= t)
                        answer_mask_array = np.array(answer_mask).T.tolist()

                    for num, answer in enumerate(answer_mask_array):
                        if answer == predict_mask and y_pred_proba_list[num] is None:
                            y_pred_oob[num] = y_pred_proba
                oob_prediction = np.array(y_pred_oob).reshape(-1, 2)[:, 1]
                self.oob_scores_list.append(accuracy_score(self.y[self.oob_idx[n]], oob_prediction))

            for mask in tree[0]:

                answer_mask = []
                predict_mask = [mask[i][0] for i in range(len(mask))]

                for question in mask:
                    idx = tree[1][question[1]][2]
                    t = tree[1][question[1]][3]
                    y_pred_proba = [tree[0][mask]]
                    answer_mask.append(X_test[:, idx] <= t)
                    answer_mask_array = np.array(answer_mask).T.tolist()

                for num, answer in enumerate(answer_mask_array):
                    if answer == predict_mask and y_pred_proba_list[num] is None:
                        y_pred_proba_list[num] = y_pred_proba

            prediction = np.array(y_pred_proba_list).reshape(-1, 2)[:, 1]
            y_proba = np.array(y_pred_proba_list).reshape(-1, 2)[:, 0]
            forest_proba.append(y_proba)
            forest_pred.append(prediction)

        self.forest_pred = np.array(forest_pred).T.reshape(X_test.shape[0], -1)
        self.forest_proba = np.array(forest_proba).T.reshape(X_test.shape[0], -1)
        if self.oob_vote:
            vote_mask = (self.oob_scores_list >= np.quantile(self.oob_scores_list, self.oob_vote))
            self.forest_proba = np.delete(self.forest_proba, vote_mask, axis=1)
            self.forest_pred = np.delete(self.forest_pred, vote_mask, axis=1)

        return mode(self.forest_pred, axis=1)[0].flatten(), np.mean(self.forest_proba, axis=1)
