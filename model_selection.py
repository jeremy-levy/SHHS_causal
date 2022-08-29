import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import xgboost as xg
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import PolynomialFeatures
from constants import y_column, x_columns, t_baseline

from GPS import gps_score
from dose_average_response import avg_dose_response


def check_best(best_model, current_model, best_auc, current_auc):
    if current_auc > best_auc:
        best_auc = current_auc
        best_model = current_model
    return best_model, best_auc


def evaluate_model(y_test, y_pred, model):
    model_auc = roc_auc_score(y_test, y_pred)
    # summarize scores
    print(model + ': ROC AUC=%.3f' % model_auc)
    # calculate roc curves
    lr_fpr, lr_tpr, _ = roc_curve(y_test, y_pred)
    # plot the roc curve for the model
    plt.plot(lr_fpr, lr_tpr, marker='.', label=model)
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # convert to f score

    return model_auc


def model_selection(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, np.ravel(Y), test_size=0.2)

    # No Skill model
    y_pred = [1 for _ in range(len(y_test))]
    best_model_auc = evaluate_model(y_test, y_pred, 'No Skill')
    best_model = lambda x: 0
    best_auc = 0.5

    # est_gp = find_best_features(X_train, y_train, column_names=X.columns)
    # y_pred = est_gp.predict(X_test)
    # model_auc = evaluate_model(y_test, y_pred, 'GeneticESTIMATOR')
    # best_model, best_auc = check_best(best_model, est_gp, best_auc, model_auc)

    # RandomForestRegressor
    regressor = RandomForestRegressor(random_state=0)
    grid_search = {'n_estimators': [5, 10, 20, 30, 40, 90, 100, 110, 120, 130, 140, 150, 200, 210, 220, 250],
                   'max_features': ['sqrt', 'log2'],
                   'max_depth': [int(x) for x in np.linspace(10, 110, num=11)],
                   'min_samples_split': [2, 5, 10, 15, ],
                   'min_samples_leaf': [1, 2, 4, 5, 10],
                   'bootstrap': [True, False],
                   }
    random_model = RandomizedSearchCV(estimator=regressor, param_distributions=grid_search,
                                      n_iter=200, random_state=32,
                                      n_jobs=-1, return_train_score=True, cv=3)
    random_model.fit(X_train, y_train)
    y_pred = random_model.predict(X_test)
    model_auc = evaluate_model(y_test, y_pred, 'RandomForestRegressor')
    best_model, best_auc = check_best(best_model, random_model, best_auc, model_auc)

    xgb_r = xg.XGBRegressor(objective='reg:squarederror', seed=123)
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
    random_model = RandomizedSearchCV(estimator=xgb_r, param_distributions=grid_search,
                                      n_iter=50, random_state=32,
                                      n_jobs=-1, return_train_score=True, cv=3)
    random_model.fit(X_train, y_train)
    y_pred = random_model.predict(X_test)
    model_auc = evaluate_model(y_test, y_pred, 'XGBRegressor')
    best_model, best_auc = check_best(best_model, random_model, best_auc, model_auc)

    # LinearRegression
    LinearReg = LinearRegression()
    LinearReg.fit(X_train, y_train)
    y_pred = LinearReg.predict(X_test)
    model_auc = evaluate_model(y_test, y_pred, 'LinearRegression')
    best_model, best_auc = check_best(best_model, random_model, best_auc, model_auc)

    # Logistic Regression
    LogisticRegressionModel = LogisticRegression(solver='lbfgs', max_iter=2000)
    LogisticRegressionModel.fit(X_train, y_train)
    y_pred = LogisticRegressionModel.predict_proba(X_test)[:, 1]
    model_auc = evaluate_model(y_test, y_pred, 'LogisticRegression')
    best_model, best_auc = check_best(best_model, random_model, best_auc, model_auc)

    # show the legend
    plt.legend()
    plt.show()
    # plt.savefig(name_dataset + '.png')

    return best_model


def linear_pred(X, T):
    predictors = np.column_stack([X, T])
    try:
        df = pd.DataFrame(predictors, columns=X.columns.append(T.columns))
        return df
    except:
        return predictors


def poly_pred(X, T):
    predictors = np.column_stack([X, T])

    try:
        predictors_df = pd.DataFrame(predictors, columns=X.columns.append(T.columns))
        poly = PolynomialFeatures(2)
        poly_predictors = poly.fit_transform(predictors_df)
        df = pd.DataFrame(poly_predictors, columns=poly.get_feature_names_out(input_features=predictors_df.columns))
        return df
    except:
        poly = PolynomialFeatures(2)
        poly_predictors = poly.fit_transform(predictors)
        return poly_predictors


if __name__ == "__main__":
    ds = pd.read_csv(r"new_data/data_SHHS1_dimensionality_reduction.csv")[x_columns + y_column + t_baseline].dropna()
    print(ds.loc[ds['outcome'] == 1].shape)
    print(ds.loc[ds['outcome'] == 0].shape)

    ds_1 = ds.loc[ds['outcome'] == 1].sample(n=4000)
    ds_0 = ds.loc[ds['outcome'] == 0].sample(n=4000)
    ds = pd.concat([ds_1, ds_0]).dropna()

    X = ds[x_columns]
    X = pd.get_dummies(X)

    Y = ds[y_column]
    T = ds[t_baseline]

    "Test with X, T linear features"
    predictors = linear_pred(X, T)
    print(predictors)
    model = model_selection(predictors, Y)
    avg_dose_response(X, T, Y, model, linear_pred)

    # "Test with X, T intersection and Polynomial features up to 2 degrees"
    # predictors = poly_pred(X, T)
    # model = model_selection(predictors, Y)
    # avg_dose_response(X, T, Y, model, poly_pred)


    # "Test with GPS score prior gamma"
    #
    # R_hat = gps_score(X, T, prior="Gamma")
    # predictors = poly_pred(R_hat, T)
    # model = model_selection(predictors, Y)
    # avg_dose_response(R_hat, T, Y, model,  poly_pred)
