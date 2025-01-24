import numpy as np
import pandas as pd


def sort_data(time_series):
    sorted_data = pd.DataFrame()
    for column in time_series:
        sorted_data[column] = time_series[column].sort_values(ascending=False). \
            reset_index(drop=True)
    return sorted_data


def construct_DC(time_series, sorted_data):
    DC = np.zeros(sorted_data.shape)
    for day in range(time_series.shape[1]):
        for j, value in enumerate(sorted_data[day].values):
            DC[j, day] = \
                np.sum(time_series.iloc[:, day] >= value) / time_series.shape[0] * 100
    return DC


def approximate_DC(time_series, b):
    approx_histogram = np.zeros((b, time_series.shape[0]))
    bins = np.linspace(0, 1, b, endpoint=False)
    for day in range(time_series.shape[0]):
        for b_indx, b_l in enumerate(reversed(bins)):
            approx_histogram[b_indx, day] = \
                np.sum(time_series[day, :] >= b_l)
    approx_DC = approx_histogram / time_series.shape[1]
    return approx_histogram.T, approx_DC.T


def get_discretised_duration_curve_for_ts(X, histogram_per_period):
    ntt = X.shape[0] * X.shape[1]
    L = np.sum(histogram_per_period, axis=0) / ntt
    return L


def calculate_pdf(data, bins):
    hist, _ = np.histogram(data, bins=bins, density=False)  # density=True normalizes to form a PDF
    return hist / len(data) + 1e-10


def cdf_calculation(data, bin_range):
    data = np.sort(data)[::-1]
    cdf = np.zeros(len(bin_range) - 1)
    for i in range(len(bin_range) - 1):
        cdf[i] = np.sum(data <= bin_range[i + 1]) / len(data)
    return cdf