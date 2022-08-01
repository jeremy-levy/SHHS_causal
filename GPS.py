import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
import pandas as pd


def gps_score(X, T, prior="Gaussian"):

    if prior == "Gaussian":
        dist = lambda x, y: stats.norm(x, y)

    elif prior == "Gamma":
        dist = lambda x, y: stats.gamma(x, y)

    N = len(X)
    model = LinearRegression()
    model_f = model.fit(X, T)
    mu = model.predict(X)  # expectation of each variable
    e = T - mu  # residual from the dose model
    s_hat = (e.T @ e) / (N - 1)  # variance of the distribution

    R_hat = []

    for t, a in zip(np.ravel(T), mu):
        R_hat.append(dist(a, s_hat).cdf(t)[0][0])
    return pd.DataFrame(R_hat, columns=['R'])


