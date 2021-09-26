import numpy as np

class MyDecisionTreeRegressor:
    NON_LEAF_TYPE = 'NON_LEAF'
    LEAF_TYPE = 'LEAF'

    def __init__(self,
                 min_samples_split=3,
                 max_depth=5,
                 min_impurity_split=0.0,
                 max_features=None,
                 split_on_diff=True,
                 random_state=None,
                 eps=1e-27):

        self.tree = dict()
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.min_impurity_split = min_impurity_split
        self.split_on_diff = split_on_diff
        self.eps = eps

        if max_features == 'sqrt':
            self.get_feature_ids = self.get_feature_ids_sqrt
        elif max_features == 'log2':
            self.get_feature_ids = self.get_feature_ids_log2
        elif max_features is None:
            self.get_feature_ids = self.get_feature_ids_N
        else:
            raise -1

    def get_feature_ids_sqrt(self, n_feature):
        feature_ids = np.arange(n_feature)
        np.random.shuffle(feature_ids)

        return feature_ids[:np.sqrt(n_feature)]

    def get_feature_ids_log2(self, n_feature):
        feature_ids = np.arange(n_feature)
        np.random.shuffle(feature_ids)

        return feature_ids[:np.log2(n_feature)]

    def get_feature_ids_N(self, n_feature):
        feature_ids = np.arange(n_feature)
        np.random.shuffle(feature_ids)

        return feature_ids

    @staticmethod
    def div_samples(x, y, feature_id, threshold):
        left_mask = x[:, feature_id] > threshold
        right_mask = ~left_mask
        
        return x[left_mask], x[right_mask], y[left_mask], y[right_mask]


    @staticmethod
    def calc_impurities_array(Y):
        N = Y.shape[1]

        y_squared = Y ** 2

        sum_squared_L = (y_squared[:, :N - 1]).cumsum(axis=1)
        sum_squared_R = y_squared.sum(axis=1)[:, None] - sum_squared_L

        mean_squared_L = sum_squared_L / np.arange(1, N)
        mean_squared_R = sum_squared_R / (N - np.arange(1, N))

        sum_L = (Y[:, :N - 1]).cumsum(axis=1)
        sum_R = Y.sum(axis=1)[:, None] - sum_L

        mean_L = sum_L / np.arange(1, N)
        mean_R = sum_R / (N - np.arange(1, N))


        cum_std_L = mean_squared_L - mean_L ** 2
        cum_std_R = mean_squared_R - mean_R ** 2

        impurities = cum_std_L * (np.arange(1, N) / N) + cum_std_R * ((N - np.arange(1, N)) / N)
        return impurities

    def calc_possible_border_ids(self, X):
        diff = np.diff(X.T, axis=1)
        idxs = np.argwhere(np.abs(diff) > self.eps)
        list_result_border_ids = []
        for i in range(X.shape[1]):
            pidxs = idxs[idxs[:, 0] == i][:,1]
            if pidxs.shape[0] == 0:
                pidxs = np.array([float('+inf')])
            list_result_border_ids.append(pidxs)
        return list_result_border_ids
    
    def __find_threshold_array(self, X, Y, feature_ids):
        sorted_ids = np.argsort(X[:, feature_ids], axis=0)
        sorted_X, sorted_Y = X[sorted_ids, feature_ids], Y[sorted_ids]

        if len(sorted_Y.shape) == 1:
            sorted_Y = sorted_Y[None, :]

        if len(sorted_X.shape) == 1:
            sorted_X = sorted_X[None, :]

        sorted_Y = sorted_Y.T
        impurities = self.calc_impurities_array(sorted_Y)

        if not self.split_on_diff:
            min_impurity_index = np.argmin(impurities, axis=1)
            min_impurity = impurities[np.arange(impurities.shape[0]), min_impurity_index.ravel()]
        else:
            possible_border_ids = self.calc_possible_border_ids(sorted_X)
            min_impurity_index = []
            min_impurity_index_none_ids = []

            for feature_index, feature_pbids in enumerate(possible_border_ids):
                if feature_pbids[0] == float('+inf'):
                    min_impurity_index.append(0)
                    min_impurity_index_none_ids.append(feature_index)
                    continue

                idx = np.argmin(impurities[feature_index, feature_pbids])
                min_impurity_index.append(feature_pbids[idx])

            min_impurity_index = np.array(min_impurity_index)
            min_impurity_index_none_ids = np.array(min_impurity_index_none_ids)
            min_impurity = impurities[np.arange(impurities.shape[0]), min_impurity_index.ravel()]
            if min_impurity_index_none_ids.shape[0] > 0:
                min_impurity[min_impurity_index_none_ids] = float('+inf')

        best_split_right_index = min_impurity_index + 1

        min_impurity[best_split_right_index <= self.min_samples_split] = float('+inf')
        min_impurity[best_split_right_index >= sorted_Y.shape[1] - self.min_samples_split] = float('+inf')

        best_threshold = np.mean([sorted_X[best_split_right_index - 1, np.arange(sorted_X.shape[1])],
                                        sorted_X[best_split_right_index, np.arange(sorted_X.shape[1])]], axis = 0)
        return min_impurity, best_threshold

    def __set_leaf(self, x, y, node_id):

        mean_value = y.mean()

        impurity = ((y - y.mean()) ** 2).mean()
        self.tree[node_id] = {'type': self.LEAF_TYPE,
                              'value': mean_value,
                              'obj_num': x.shape[0],
                              'impurity': impurity
                               }

    def __check_is_leaf(self, X, Y, depth):

        if X.shape[0] < 2 * self.min_samples_split + 2 or np.unique(Y).shape[0] == 1:
            return True

        if self.max_depth is not None and depth >= self.max_depth:
            return True

        if self.min_impurity_split is not None and np.mean((Y - Y.mean()) ** 2) < self.min_impurity_split:
            return True

        return  False


    def __fit_node(self, X, Y, node_id, depth):
        
        if self.__check_is_leaf(X, Y, depth):
            self.__set_leaf(X, Y, node_id)
            return

        node_impurity = ((Y - Y.mean()) ** 2).mean()

        feature_ids = self.get_feature_ids(X.shape[1])

        th_results = self.__find_threshold_array(X, Y, feature_ids)
        impurities = th_results[0]
        thresholds = th_results[1]

        best_impurity_index = np.argmin(impurities)
        best_impurity = impurities[best_impurity_index]

        best_treshold = thresholds[best_impurity_index]
        best_feature_id = feature_ids[best_impurity_index]

        if best_impurity == float('+inf'):
            self.__set_leaf(X, Y, node_id)
            return

        left_x, right_x, left_y, right_y = self.div_samples(X, Y, best_feature_id, best_treshold)

        if left_x.shape[0] == 0 or right_x.shape[0] == 0:
            self.__set_leaf(X, Y, node_id)
            return

        self.tree[node_id] = {'type': self.NON_LEAF_TYPE,
                              'feature_id': best_feature_id,
                              'threshold': best_treshold,
                              'impurity': node_impurity,
                              'obj_num': X.shape[0]}

        self.__fit_node(left_x, left_y, 2 * node_id + 1, depth + 1)
        self.__fit_node(right_x, right_y, 2 * node_id + 2, depth + 1)

        return

    def fit(self, X, Y):
        self.__fit_node(X, Y, 0, 0)

    def __predict_value(self, x, node_id):
        node = self.tree[node_id]

        if node['type'] == self.__class__.NON_LEAF_TYPE:
            feature_id, threshold = node['feature_id'], node['threshold']
            if x[feature_id] > threshold:
                return self.__predict_value(x, 2 * node_id + 1)
            else:
                return self.__predict_value(x, 2 * node_id + 2)
        else:
            return node['value']

    def predict(self, X):
        return np.array([self.__predict_value(x, 0) for x in X])

    def get_leaf_values(self):
        res = []
        for node in self.tree:
            if self.tree[node]['type'] == self.LEAF_TYPE:
                res.append(self.tree[node]['value'])
        return res

class MyMeanEstimator:
    def __init():
        pass

    def fit(self, X, y):
        self.tmp = np.mean(y)

    def predict(self, X):
        pred = [self.tmp for i in range(X.shape[0])]
        return np.array(pred)

class MyGradientBoostingRegressor:
    def __init__(self, 
                 max_depth=3, 
                 n_estimators=300,
                 base_estimator=MyDecisionTreeRegressor,
                 min_samples_split=2, 
                 lr=0.1):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.min_samples_split = min_samples_split
        self.lr = lr
        self.models = [None] * (self.n_estimators + 1)

    @staticmethod    
    def get_loss(y_pred, y_true):
        return ((y_pred - y_true) ** 2).mean()
    
    @staticmethod
    def get_antigrad(y_pred, y_true):
        return 2.0 * (- y_pred + y_true)
    
    @staticmethod
    def get_b(y_true, h, a):
        return (np.sum((y_true - h) * a)) / (np.sum(a ** 2)) 
    
    def fit(self, X, y):
        self.bs = np.zeros(self.n_estimators + 1)
        self.models[0] = MyMeanEstimator()
        self.bs[0] = 1
        self.models[0].fit(X, y)
        pred = self.models[0].predict(X) * self.bs[0]
        curr_depth = self.max_depth
        for i in range(1, self.n_estimators + 1):
            antigrad = self.get_antigrad(pred, y)
            self.models[i] = self.base_estimator(min_samples_split=self.min_samples_split, max_depth=curr_depth)
            self.models[i].fit(X,antigrad)
            local_pred = self.models[i].predict(X)
            self.bs[i] = self.lr * self.get_b(y, pred, local_pred)
            pred = pred + self.bs[i] * local_pred
        return self
    
    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i in range(self.n_estimators + 1):
            y_pred = y_pred + self.bs[i] * self.models[i].predict(X)
        return y_pred

    def staged_predict(self, X):
        y_pred = np.zeros(X.shape[0])
        y_pred_array = []
        for i in range(self.n_estimators + 1):
            y_pred = y_pred + self.bs[i] * self.models[i].predict(X)
            y_pred_array.append(y_pred)
        return y_pred_array[1:]

class MyBaggingRegressor:
    def __init__(self,
                  base_estimator=None, 
                  n_estimators=10, 
                  max_samples=1.0, 
                  max_features=1.0, 
                  bootstrap=False, 
                  bootstrap_features=False):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.X_masks = None
        self.f_masks = None
        self.models = [None] * (self.n_estimators)
    
    def get_subsample_F(self, X_shape, replace_):
        mask_ids = np.random.choice(np.arange(X_shape[1]), size=self.subsample_F_size, replace=replace_)
        return np.sort(mask_ids)
    
    def get_subsample(self, X, y, replace_):
        mask_ids = np.random.choice(np.arange(X.shape[0]), size=self.subsample_size, replace=replace_)
        return X[mask_ids], y[mask_ids]

    def transf_subsamples(self, X_masks, f_masks):
        self.X_masks = X_masks
        self.f_masks = f_masks
    
    def fit(self, X, y):
        if self.X_masks is not None and self.f_masks is not None:

            self.mask = self.f_masks

            for i in range(self.n_estimators):
                model = self.base_estimator
                X_subsampled = X[:, self.f_masks[i]]
                X_subsampled, y_subsampled = X_subsampled[self.X_masks[i]], y[self.X_masks[i]]
                model.fit(X_subsampled, y_subsampled)
                self.models[i] = model
        else:     
            self.subsample_size = int(X.shape[0] * self.max_samples)
            self.subsample_F_size = int(X.shape[1] * self.max_features)

            self.mask = []
        
            for i in range(self.n_estimators):
                model = self.base_estimator
                self.mask.append(self.get_subsample_F(X.shape, self.bootstrap_features))
                X_subsampled = X[:, self.mask[i]]
                X_subsampled, y_subsampled = self.get_subsample(X_subsampled, y, self.bootstrap)
                model.fit(X_subsampled, y_subsampled)
                self.models[i] = model

        return self
    
    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
    
        for i in range(self.n_estimators):
            y_pred = y_pred + self.models[i].predict(X[:, self.mask[i]])
        return y_pred / (self.n_estimators)
    
    def predict_array(self, X):
        y_array = []
        y_pred = np.zeros(X.shape[0])
        for i in range(self.n_estimators):
            y_pred = y_pred + self.models[i].predict(X[:, self.mask[i]])
            y_array.append(y_pred / (i + 1))
        return y_array