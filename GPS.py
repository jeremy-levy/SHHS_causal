import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
import pandas as pd
from tqdm import tqdm


def gps_score(X, T, model, prior="Gaussian"):
    T = np.ravel(T)

    if prior == "Gaussian":
        dist = stats.norm

    elif prior == "Gamma":
        dist = stats.gamma

    else:
        return

    N = len(X)

    # model = LinearRegression()
    model.fit(X, T)
    mu = model.predict(X)  # expectation of each variable
    e = T - mu  # residual from the dose model
    s_hat = (e.T @ e) / (N - 1)  # variance of the distribution

    R_hat = []

    for t, a in zip(T, mu):
        R_hat.append(dist(a, s_hat).cdf(t))
    return np.array(R_hat)
