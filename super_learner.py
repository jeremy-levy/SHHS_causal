import sys

sys.path.append('/home/jeremy.levy/Jeremy/copd_osa')

import xgboost as xg
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import pandas as pd
import os
from icecream import ic
from category_encoders.cat_boost import CatBoostEncoder
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor

from SHHS_Causal.constants import t_baseline, x_columns_categorical, x_columns_continuous, y_column, t_autoencoder, \
    t_pca
from SHHS_Causal.dose_average_response import avg_dose_response
import utils.graphics as graph
from SHHS_Causal.GPS import gps_score


def bootstrap_one_iteration(X, Y, T, min_T, max_T, delta_t, model, include_gps):
    if include_gps is True:
        R_hat = gps_score(X, T, model=model, prior="Gamma")
        data = np.column_stack([R_hat, X, T])
        mu_data = np.column_stack([R_hat, X])
    else:
        data = np.column_stack([X, T])
        mu_data = X

    Y = Y[~np.isnan(data[:, 0])]
    data = data[~np.isnan(data[:, 0])]
    mu_data = mu_data[~np.isnan(mu_data[:, 0])]

    data_train, data_test, y_train, y_test = train_test_split(data, np.ravel(Y), test_size=0.2)

    model.fit(data_train, y_train)
    y_pred = model.predict(data_test)

    model_auc = roc_auc_score(y_test, y_pred)
    mu_t = avg_dose_response(mu_data, min_T, max_T, delta_t, model)

    return model_auc, mu_t


def get_model(model, n_jobs):
    if model == 'xgb':
        base_model = xg.XGBRegressor(objective='reg:squarederror', seed=123)
        grid_search = {"n_estimators": np.linspace(10, 150, 10, dtype=int),
                       "max_depth": np.linspace(2, 15, 10, dtype=int),
                       "learning_rate": np.logspace(-5, 0, 15),
                       "reg_alpha": np.logspace(-5, 0, 5),
                       "reg_lambda": np.logspace(-5, 0, 5),
                       'min_child_weight': [1, 5, 10],
                       'gamma': [0.5, 1, 1.5, 2, 5],
                       'subsample': [0.6, 0.8, 1.0],
                       'colsample_bytree': [0.6, 0.8, 1.0],
                       }
    elif model == 'rf':
        base_model = RandomForestRegressor(random_state=0)
        grid_search = {'n_estimators': [5, 10, 20, 30, 40, 90, 100, 110, 120, 130, 140, 150, 200, 210, 220, 250],
                       'max_features': ['sqrt', 'log2'],
                       'max_depth': [int(x) for x in np.linspace(10, 110, num=11)],
                       'min_samples_split': [2, 5, 10, 15, ],
                       'min_samples_leaf': [1, 2, 4, 5, 10],
                       'bootstrap': [True, False],
                       }
    else:
        return

    model_search = RandomizedSearchCV(estimator=base_model, param_distributions=grid_search,
                                      n_iter=50, random_state=32,
                                      n_jobs=n_jobs, return_train_score=True, cv=3)
    return model_search


def bootstrap(n_jobs, B, include_gps, t_column_chosen, str_model, delta_t=0.1):
    model = get_model(model=str_model, n_jobs=n_jobs)

    all_data = pd.read_csv(os.path.join('/home/jeremy.levy/Jeremy/copd_osa/SHHS_Causal/',
                                        'data_dimensionality_reduction.csv'))
    all_T = all_data[t_column_chosen].values
    min_T, max_T = np.min(all_T), np.max(all_T)

    all_model_auc, all_mu_t = [], []
    for i in tqdm(range(B)):
        data_b = all_data.sample(5000)

        T = data_b[t_column_chosen].values
        Y = data_b[y_column].values

        X_continuous = data_b[x_columns_continuous].values
        X_categorical = data_b[x_columns_categorical].values
        X_categorical = CatBoostEncoder().fit_transform(X_categorical, Y).values
        X = np.concatenate([X_continuous, X_categorical], axis=1)

        model_auc, mu_t = bootstrap_one_iteration(X, Y, T, min_T, max_T, delta_t, model, include_gps=include_gps)
        all_model_auc.append(model_auc)
        all_mu_t.append(mu_t)

    all_model_auc = np.array(all_model_auc)
    print('AUC: ', np.median(all_model_auc), '+/-', np.std(all_model_auc))

    all_mu_t = np.array(all_mu_t)
    filename = 'mu_t_' + t_column_chosen[0] + '_gps_' + str(include_gps) + '_' + str_model + '.npy'
    np.save(os.path.join('/home/jeremy.levy/Jeremy/copd_osa/SHHS_Causal/save_np', filename), all_mu_t)


def figure_bootstrap():
    base_path = '/home/jeremy.levy/Jeremy/copd_osa/SHHS_Causal/save_np'

    def one_panel(filename, ax, min_T, max_T):
        ic(min_T, max_T)
        all_mu_t = np.load(os.path.join(base_path, filename))
        mean_mu_t = np.mean(all_mu_t, axis=0)
        std_mu_t = np.std(all_mu_t, axis=0)

        x_coordinate = np.arange(start=min_T, stop=max_T + 0.1, step=0.1)

        ax.plot(x_coordinate, mean_mu_t, c='dodgerblue')
        ax.fill_between(x=x_coordinate, y1=mean_mu_t - std_mu_t, y2=mean_mu_t + std_mu_t, color='lightskyblue',
                        alpha=0.5)

    all_data = pd.read_csv(os.path.join('/home/jeremy.levy/Jeremy/copd_osa/SHHS_Causal/',
                                        'data_dimensionality_reduction.csv'))

    fig, axes = graph.create_figure(subplots=(3, 2), figsize=(16, 24))
    ticks_fontsize, fontsize, letter_fontsize = 15, 15, 15

    all_T = all_data[t_baseline].values
    min_T_baseline, max_T_baseline = np.min(all_T), np.max(all_T)
    one_panel(filename='mu_t_CA_relative_3_gps_False_rf.npy', ax=axes[0][0], min_T=min_T_baseline, max_T=max_T_baseline)
    one_panel(filename='mu_t_CA_relative_3_gps_True_rf.npy', ax=axes[0][1], min_T=min_T_baseline, max_T=max_T_baseline)

    all_T = all_data[t_autoencoder].values
    min_T_autoencoder, max_T_autoencoder = np.min(all_T), np.max(all_T)
    one_panel(filename='mu_t_autoencoder_treatment_gps_False_rf.npy', ax=axes[1][0], min_T=min_T_autoencoder, max_T=max_T_autoencoder)
    one_panel(filename='mu_t_autoencoder_treatment_gps_True_rf.npy', ax=axes[1][1], min_T=min_T_autoencoder, max_T=max_T_autoencoder)

    all_T = all_data[t_pca].values
    min_T_pca, max_T_pca = np.min(all_T), np.max(all_T)
    one_panel(filename='mu_t_pca_treatment_gps_False_rf.npy', ax=axes[2][0], min_T=min_T_pca, max_T=max_T_pca)
    one_panel(filename='mu_t_pca_treatment_gps_True_rf.npy', ax=axes[2][1], min_T=min_T_pca, max_T=max_T_pca)

    x_pos, y_pos = -0.1, 1.01
    axes[0][0].text(x_pos, y_pos, "(a)", fontsize=letter_fontsize, transform=axes[0][0].transAxes)
    axes[0][1].text(x_pos, y_pos, "(b)", fontsize=letter_fontsize, transform=axes[0][1].transAxes)
    axes[1][0].text(x_pos, y_pos, "(c)", fontsize=letter_fontsize, transform=axes[1][0].transAxes)
    axes[1][1].text(x_pos, y_pos, "(d)", fontsize=letter_fontsize, transform=axes[1][1].transAxes)
    axes[2][0].text(x_pos, y_pos, "(e)", fontsize=letter_fontsize, transform=axes[2][0].transAxes)
    axes[2][1].text(x_pos, y_pos, "(f)", fontsize=letter_fontsize, transform=axes[2][1].transAxes)

    graph.complete_figure(fig, axes, put_legend=[[False, False], [False, False], [False, False]],
                          y_titles=[['Average dose function', ''], ['Average dose function', ''],
                                    ['Average dose function', '']],
                          x_titles=[['$CA_{3}$', '$CA_{3}$'], ['Auto-encoder', 'Auto-encoder'], ['PCA', 'PCA']],
                          xticks_fontsize=ticks_fontsize, yticks_fontsize=ticks_fontsize,
                          xlabel_fontsize=fontsize, ylabel_fontsize=fontsize, tight_layout=True,
                          savefig=True, main_title='bootstrap_super_learner',
                          legend_fontsize=fontsize)


if __name__ == "__main__":
    # bootstrap(n_jobs=-1, B=1500, include_gps=False, str_model='rf', t_column_chosen=t_baseline)
    # bootstrap(n_jobs=-1, B=1500, include_gps=True, str_model='rf', t_column_chosen=t_baseline)
    # bootstrap(n_jobs=-1, B=1500, include_gps=False, str_model='rf', t_column_chosen=t_autoencoder)
    # bootstrap(n_jobs=-1, B=1500, include_gps=True, str_model='rf', t_column_chosen=t_autoencoder)
    # bootstrap(n_jobs=-1, B=1500, include_gps=False, str_model='rf', t_column_chosen=t_pca)
    # bootstrap(n_jobs=-1, B=1500, include_gps=True, str_model='rf', t_column_chosen=t_pca)

    figure_bootstrap()
