from Optimization.opt_utils import *
from Clustering.cluster_utils import euclidean_distance

import cvxpy as cp


class dc_opt:
    def __init__(self, bin_number=10, Nrepr=5):
        self.B = bin_number
        self.K = Nrepr
        self.u = None
        self.w = None
        self.N = None
        self.optimization_result = None
        self.Lb = None
        self.Abd = None
        self.labels = None

    def define_optimization_problem(self, X):
        """
        Defines the optimization problem based on the processed data.
        """

        # Defining parameters
        self.N = X.shape[0]
        histogram, self.Abd = approximate_DC(X, b=self.B)
        self.Lb = get_discretised_duration_curve_for_ts(X, histogram).reshape(1, -1)
        # Defining variables
        num_days = self.N
        self.u = cp.Variable(num_days, boolean=True)
        self.w = cp.Variable(num_days)

        # Objective function
        A = self.Abd - self.Lb
        objective = cp.Minimize(cp.sum_squares(A.T @ (self.w / self.N)))

        # Constraints
        constraints = [
            cp.sum(self.u) == self.K,
            self.w <= self.u * num_days,
            self.w >= 0,
            cp.sum(self.w) == num_days
        ]

        # Define the problem
        problem = cp.Problem(objective, constraints)

        return problem

    def solve_problem(self, problem):
        problem.solve()
        self.optimization_result = problem
        self.labels = self.__assign_to_RPs()

    def get_results(self):
        if self.optimization_result is None:
            raise ValueError("The optimization problem has not been solved yet.")

        return {
            "status": self.optimization_result.status,
            "optimal_value": self.optimization_result.value,
        }

    def get_RPs_indx(self):
        return np.where(self.u.value > 0.5)[0]

    def get_RPs(self, X):
        RPs = {}
        RPs_index = self.get_RPs_indx()
        for index in RPs_index:
            RPs.update({index: X[index]})
        RP_df = pd.DataFrame.from_dict(RPs)
        return RP_df.T

    def get_weights(self):
        # N = X.shape[0]
        weights = self.w.value[np.where(self.w.value > 0.0)] / self.N
        return weights.reshape(-1, 1)

    def __assign_to_RPs(self):
        R = np.zeros((self.N, self.K))
        centers = self.Abd[self.get_RPs_indx()]
        for ind, ts_i in enumerate(self.Abd):
            min_dist = float('inf')
            closest_clust = None
            for c_ind, c_ts in enumerate(centers):
                c_array = np.array(c_ts)
                cur_dist = euclidean_distance(ts_i, c_array)
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    closest_clust = c_ind
            R[ind, closest_clust] = 1
        return R
