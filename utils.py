import os
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from collections import defaultdict

from Optimization.methods import dc_opt
from Optimization.opt_utils import approximate_DC
from Clustering.methods import Kmedoids


def get_csv_files_in_directory(directory: str, pattern: str = None):
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    if pattern:
        files = [f for f in files if pattern in f]
    return files


def read_input_data(data_path: str, delimiter: str = ',', header=None):
    df = pd.read_csv(data_path, header=header, delimiter=delimiter)
    df = df.melt(var_name='original_column', value_name='data')['data'].to_frame()
    return df


def load_customer_profiles(load_path):
    files = get_csv_files_in_directory(load_path, pattern='Load')
    customers_profile = {}
    annual_e = {}
    for file in files:
        customer_id = file.split('.')[0]
        selected_file = os.path.join(load_path, file)
        energy_data = read_input_data(selected_file).values
        customers_profile[customer_id] = energy_data
        annual_e[customer_id] = np.sum(energy_data)
    return customers_profile, annual_e


def classify_customers(annual_e, thresholds):
    categories = {"small": [], "medium": [], "large": []}
    for customer, consumption in annual_e.items():
        if -thresholds["small"] <= consumption <= thresholds["small"]:
            categories["small"].append(customer)
        elif -thresholds["medium"] <= consumption <= thresholds["medium"]:
            categories["medium"].append(customer)
        else:
            categories["large"].append(customer)
    return categories


def select_customers(categories):
    return {
        category: random.choice(customers) if customers else None
        for category, customers in categories.items()
    }


def reshape_data(data, time_step_minutes):
    # Calculate number of time steps per day (24 hours * 60 minutes / time_step_minutes)
    time_steps_per_day = (24 * 60) // time_step_minutes

    # Validate that the data can be reshaped
    total_time_steps = len(data)
    if total_time_steps % time_steps_per_day != 0:
        raise ValueError(
            f"Data length ({total_time_steps}) is not divisible by the number of time steps per day ({time_steps_per_day})."
        )

    # Calculate number of days
    num_days = total_time_steps // time_steps_per_day

    # Reshape the data
    reshaped_array = data.reshape(num_days, time_steps_per_day)

    return reshaped_array


def normalize_data(data):
    data = data * 4
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return reshape_data(normalized_data, time_step_minutes=15)


def fast_error_calculation(X, cluster_labels, RPs, mode='cluster'):
    X = np.array(X)
    cluster_labels = np.array(cluster_labels)
    RPs = np.array(RPs)

    t_max, t_min = np.max(X), np.min(X)
    range_t = np.abs(t_max - t_min)
    HR = X.size  # Total number of elements

    if mode == 'cluster':
        # Broadcasting to calculate squared differences
        squared_differences = ((X[:, None, :] - RPs[None, :, :]) ** 2) * cluster_labels[:, :, None]
        cluster_error_sum = np.sum(squared_differences)

    elif mode == 'optimization':
        # Calculate squared differences for optimization mode
        squared_differences = (X - RPs) ** 2
        cluster_error_sum = np.sum(squared_differences)

    else:
        raise ValueError("Invalid mode. Choose 'cluster' or 'optimization'.")

    # Normalize and compute RMSE
    normalized_sum = cluster_error_sum / HR
    RMSE = np.sqrt(normalized_sum) * (1 / range_t)
    return RMSE


def calculate_rldce(dc, dc_hat):
    abs_errors = np.abs((np.sum(dc, axis=1) - np.sum(dc_hat)) / np.sum(dc, axis=1))
    rldc_error = np.mean(abs_errors)
    return rldc_error


def calculate_exerror(data, rps):
    maxerror = (np.abs(np.max(data) - np.max(rps)) / np.max(data))
    # minerror = (np.abs(np.min(data) - np.min(rps)) / np.min(data)) * 100
    return maxerror


def perform_clustering(clustering_methods, n_epoch, selected_customers, customers_profile):
    results = {label: {
        'RMSE': defaultdict(list),
        'REEav': defaultdict(list),
        'RELE': defaultdict(list),
    } for label in selected_customers.keys()}

    for i in range(n_epoch):
        for label, c_id in selected_customers.items():
            if c_id is not None:
                input_data = normalize_data(customers_profile[c_id])
                _, dcs = approximate_DC(input_data, b=10)
                for method_name, cluster_function in clustering_methods.items():
                    cluster = cluster_function()
                    if isinstance(cluster, dc_opt):
                        if i == 0:
                            opt_problem = cluster.define_optimization_problem(input_data)
                            cluster.solve_problem(opt_problem)
                            RPs = cluster.get_RPs(input_data)
                            weights = cluster.get_weights()
                            try:
                                rep_dc = np.sum(dcs[RPs.index] * weights, axis=0)
                            except ValueError as e:
                                print(
                                    f"Skipping computation for {method_name} in category {label} due to shape mismatch: {e}")
                                continue
                            rmse = fast_error_calculation(input_data, cluster.labels, RPs.values)
                            max_e = calculate_exerror(input_data, RPs.values)
                            rdce = calculate_rldce(dcs, rep_dc)
                            results[label]['RMSE'][method_name].append(rmse)
                            results[label]['REEav'][method_name].append(rdce)
                            results[label]['RELE'][method_name].append(max_e)
                    else:
                        if isinstance(cluster, Kmedoids):
                            if cluster.distance_metric == 'fast-dtw':
                                if i == 0:
                                    dist_dtw = cluster.dist_matrix_calculation(input_data)
                                cluster.fit(input_data, dist_matrix=dist_dtw)
                            elif cluster.distance_metric == 'euclidean':
                                dist = cluster.dist_matrix_calculation(input_data)
                                cluster.fit(input_data, dist_matrix=dist)
                        else:
                            cluster.fit(input_data)

                        RPs = cluster.get_RPs(input_data)
                        RPs_idxs = RPs.index
                        RPs = RPs.values
                        weights = cluster.get_weights()
                        try:
                            rep_dc = np.sum(dcs[RPs_idxs] * weights, axis=0)
                        except ValueError as e:
                            print(
                                f"Skipping computation for {method_name} in category {label} due to shape mismatch: {e}")
                            continue
                        rmse = fast_error_calculation(input_data, cluster.labels, RPs)
                        rdce = calculate_rldce(dcs, rep_dc)
                        max_e = calculate_exerror(input_data, RPs)
                        results[label]['RMSE'][method_name].append(rmse)
                        results[label]['REEav'][method_name].append(rdce)
                        results[label]['RELE'][method_name].append(max_e)
        print(f"Epoch {i + 1}/{n_epoch} completed")
    return results


def plot_all_categories(results, metric):
    categories = results.keys()
    method_names = list(next(iter(results.values()))[metric].keys())
    x = np.arange(len(method_names))
    width = 0.1  # Bar width
    category_colors = {
        "small": "blue",
        "medium": "green",
        "large": "orange"
    }

    plt.style.use(['science', 'no-latex'])
    plt.figure(figsize=(8, 3))

    for i, category in enumerate(categories):
        metric_data = {
            method: results[category][metric][method] if method in results[category][metric] else []
            for method in method_names
        }

        # Plot boxplot for non-dc-opt_model methods
        for j, (method, data) in enumerate(metric_data.items()):
            if method != "ldc-opt":
                plt.boxplot(
                    data,
                    positions=[j + i * width],
                    widths=0.1,
                    patch_artist=True,
                    boxprops=dict(facecolor=category_colors[category], alpha=0.5),
                    medianprops=dict(color='black')
                )

            # Special handling for "dc-opt_model" values
            elif method == "ldc-opt" and data:
                star_value = data[0]  # Assuming one value for dc-opt_model
                plt.scatter(j + i * width, star_value, color=category_colors[category], marker='*', s=100,
                            label=f"{category.capitalize()}")

    plt.xticks(x + width, method_names, fontsize=10)
    plt.xlabel("Clustering Methods", fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    plt.savefig(f"{metric}-results.pdf", format="pdf", pad_inches=0, bbox_inches="tight", transparent=False)
