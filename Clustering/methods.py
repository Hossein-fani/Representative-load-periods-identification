from Clustering.cluster_utils import *
import pandas as pd


class Cluster:
    def __init__(self, n_clusters, max_iter, distance_metric='euclidean', w=10, init_method='random', extreme=False):
        self.k = n_clusters
        self.max_iter = max_iter
        # self.tol = tol
        self.distance_metric = distance_metric.lower()
        self.init_method = init_method.lower()
        self.centers = None
        self.labels = None
        self.costs = []
        self.w = w
        self.extreme = extreme

        # Map of supported distance metrics
        self.distance_functions = {
            "euclidean": euclidean_distance,
            "dtw": dtw_distance,
            "fast-dtw": fast_dtw_Distance,
        }

        # Validate distance metric
        if self.distance_metric not in self.distance_functions:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

    def compute_distance(self, x, y):
        if self.distance_metric == "fast-dtw":
            return self.distance_functions[self.distance_metric](x, y, w=self.w)
        else:
            return self.distance_functions[self.distance_metric](x, y)

    def fit(self, X):
        raise NotImplementedError("The 'fit' method must be implemented in subclasses.")

    def initialize_centroids(self, X):
        if self.init_method == "random":
            # Random initialization
            rand_ind = np.random.choice(len(X), self.k, replace=False)
            return X[rand_ind], rand_ind
        elif self.init_method == "kmean++":
            return self._greedy_initiliaze(X)

    def _greedy_initiliaze(self, X):
        N, D = X.shape
        rand_inds = []
        rand_ind = np.random.choice(N)
        rand_inds.append(rand_ind)
        C = [X[rand_ind]]  # Centroids list starts with a single random point
        for i in range(1, self.k):
            # Calculate distances to the closest centroid for each data point
            dist = np.array([min(self.compute_distance(x, c) for c in C) for x in X])
            dist_f = dist ** 2
            # Compute probabilities for selecting the next centroid
            prob = dist_f / np.sum(dist_f)
            # Select the next centroid based on the probabilities
            rand_ind = np.random.choice(N, p=prob)
            next_centroid = X[rand_ind]
            rand_inds.append(rand_ind)
            C.append(next_centroid)
        rand_inds = np.array(rand_inds)
        C = np.array(C)
        return C, rand_inds

    def get_weights(self):
        weights = np.array(
            [sum(self.labels[:, key]) / sum(sum(self.labels[:, key]) for key in range(self.k))
             for key in range(self.k)])
        return weights.reshape(-1, 1)

    def get_RPs(self, X):
        RPs = {}
        if isinstance(self, Kmedoids):
            for k in range(self.k):
                ind = self.curr_medoids[k]
                RPs.update({ind: X[ind]})
            RP_df = pd.DataFrame.from_dict(RPs)
            return RP_df.T
        else:
            for k, center in enumerate(self.centers):
                diff = []
                cluster_indx = np.where(self.labels[:, k] >= 0.5)[0]
                for ind in cluster_indx:
                    diff.append(self.compute_distance(X[ind], center))
                min_index = cluster_indx[np.argmin(np.array(diff))]
                RPs.update({min_index: X[min_index]})
            RP_df = pd.DataFrame.from_dict(RPs)
            return RP_df.T

    def _cost_calculation(self, X):
        cost = 0
        for k in range(self.k):
            diff = X - self.centers[k]
            sq_distances = (diff * diff).sum(axis=1)
            cost += (self.labels[:, k] * sq_distances).sum()
        return cost


class Kmeans(Cluster):
    def __init__(self, n_clusters, max_iter, distance_metric='euclidean', w=10, init_method='random', extreme=False):
        super().__init__(n_clusters, max_iter, distance_metric, w, init_method, extreme)

    def __assign_to_cluster(self, X):
        # assignments = {}
        N, D = X.shape
        R = np.zeros((N, self.k))
        for ind, ts_i in enumerate(X):
            min_dist = float('inf')
            closest_clust = None
            for c_ind, c_ts in enumerate(self.centers):
                c_array = np.array(c_ts)
                # if c_ind not in assignments:
                #     assignments[c_ind] = []
                cur_dist = self.compute_distance(ts_i, c_array)
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    closest_clust = c_ind
            R[ind, closest_clust] = 1
            # assignments[closest_clust].append(ind)
        return R

    def __update_centroids(self, X):
        centers = self.centers
        for key in range(self.k):
            clust_sum = 0
            cluster_indx = np.where(self.labels[:, key] == 1)[0]
            for ind in cluster_indx:
                clust_sum += X[ind]
            # centers[key] = [m / len(cluster_indx) for m in clust_sum]
            try:
                centers[key] = [m / len(cluster_indx) for m in clust_sum]
            except TypeError:
                print(f"Error assigning centers for key {key}. Skipping.")
        return centers

    def fit(self, X):
        self.centers, _ = self.initialize_centroids(X)
        for n in range(self.max_iter):
            self.labels = self.__assign_to_cluster(X)
            self.centers = self.__update_centroids(X)
            cost = self._cost_calculation(X)
            self.costs.append(cost)

            if n > 0:
                if np.abs(self.costs[-1] - self.costs[-2]) < 1e-5:
                    break

            if len(self.costs) > 1:
                if self.costs[-1] > self.costs[-2]:
                    pass


class Kmedoids(Cluster):
    def __init__(self, n_clusters, max_iter, distance_metric='euclidean', w=10, init_method='random', extreme=False):
        super().__init__(n_clusters, max_iter, distance_metric, w, init_method, extreme)
        self.curr_medoids = np.array([-1] * self.k)

    def dist_matrix_calculation(self, X):
        N, D = X.shape
        dist_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                dist_matrix[i, j] = self.compute_distance(X[i, :], X[j, :])
            if self.distance_metric == 'fast-dtw':
                print(f'Distance {i} calculated')

        return dist_matrix

    def __assign_to_clusters(self, dist_matrix):
        N, N = dist_matrix.shape
        R = np.zeros((N, self.k))
        distance_to_medoids = dist_matrix[:, self.curr_medoids]
        assignment = np.argmin(distance_to_medoids, axis=1)
        for ind, label in enumerate(assignment):
            R[ind, label] = 1
        clusters = self.curr_medoids[np.argmin(distance_to_medoids, axis=1)]
        clusters[self.curr_medoids] = self.curr_medoids
        return R

    def __update_medoids(self, dist_matrix):
        new_medoids = np.array([-1] * self.k)
        for k, curr_medoid in enumerate(self.curr_medoids):
            cluster = np.where(self.labels[:, k] == 1)[0]
            mask = np.ones(dist_matrix.shape)
            mask[np.ix_(cluster, cluster)] = 0.
            cluster_distances = np.ma.masked_array(data=dist_matrix, mask=mask, fill_value=10e9)
            costs = cluster_distances.sum(axis=1)
            new_medoids[self.curr_medoids == curr_medoid] = costs.argmin(axis=0, fill_value=10e9)
        return new_medoids

    def fit(self, X, dist_matrix=None, plot_cost=False):
        counter = 0
        N, D = X.shape
        old_medoids = np.array([-1] * self.k)
        _, self.curr_medoids = self.initialize_centroids(X)
        # dist_matrix = self.__dist_matrix_calculation(X)
        while not ((old_medoids == self.curr_medoids).all()):
            self.labels = self.__assign_to_clusters(dist_matrix)
            old_medoids = self.curr_medoids
            self.curr_medoids = self.__update_medoids(dist_matrix)
            counter += 1
            # print(counter)
        self.centers = X[self.curr_medoids]

    def calculate_extremes(self, distance_matrix):
        extremes = []
        for k, medoid in enumerate(self.curr_medoids):
            cluster = np.where(self.labels[:, k] == 1)[0]
            cluster = cluster[cluster != medoid]
            masked_matrix = np.zeros_like(distance_matrix)
            for r in self.curr_medoids:
                for c in cluster:
                    masked_matrix[r, c] = distance_matrix[r, c]
            extremes.append(np.argmax(np.sum(masked_matrix, axis=0)))
        return extremes
