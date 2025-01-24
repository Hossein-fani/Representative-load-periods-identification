from Clustering.methods import Kmeans
from utils import *
import constants

import random
from functools import partial


random.seed(40)

Nrep = 5

load_path = constants.data_path1

thresholds = {
    "small": 600,
    "medium": 3500,
    "large": 7500
}

customers_profile, annual_e = load_customer_profiles(load_path)
categories = classify_customers(annual_e, thresholds)
selected_customers = select_customers(categories)


clustering_methods = {
    "K-means": partial(Kmeans, n_clusters=Nrep, max_iter=20, distance_metric='euclidean'),
    "K-medoids": partial(Kmedoids, n_clusters=Nrep, max_iter=20, distance_metric='euclidean'),
    "K-medoids_dtw": partial(Kmedoids, n_clusters=Nrep, max_iter=20, distance_metric='fast-dtw', w=10),
    # "K-medoids_e": partial(Kmedoids, n_clusters=Nrep, max_iter=20, distance_metric='euclidean', extreme=True),
    # "K-medoids_dtw_e": partial(Kmedoids, n_clusters=Nrep, max_iter=20, distance_metric='fast-dtw', w=10, extreme=True),
    "ldc-opt": partial(dc_opt, Nrepr=Nrep)
}

n_epoch = 50
results = perform_clustering(clustering_methods, n_epoch, selected_customers, customers_profile)
plot_all_categories(results, "RMSE")
plot_all_categories(results, "REEav")
plot_all_categories(results, "RELE")
