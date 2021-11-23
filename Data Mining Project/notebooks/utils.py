import os
import pathlib
import pandas as pd

from datetime import datetime
from dateutil.relativedelta import relativedelta

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import Pipeline


def read_to_df(filename):
    load_path = pathlib.Path().joinpath("..").joinpath("data").joinpath(filename)
    return pd.read_csv(load_path, delimiter=";")


def read_date(date):
    date = str(date)

    return datetime.strptime(f"{int(date[:2]) + 1900}/{int(date[2:4]) % 50}/{date[4:6]}", "%Y/%m/%d")


def calculate_age(birth_date):
    end_date = datetime(1999, 1, 1)

    return relativedelta(end_date, birth_date).years


def get_X_y(dataset, columns_to_drop, target_column, scaler=None):
    X = dataset.drop(columns_to_drop, axis=1)
    y = dataset[target_column]

    if scaler != None:
        scaler = scaler.fit(X)
        X = scaler.transform(X)

    return X, y


def tune_model(dataset, model_instance, parameter_grid,  target_column, cross_validation=StratifiedKFold(n_splits=10), scaler=None):
    X, y = get_X_y(dataset, columns_to_drop, target_column, scaler)

    grid_search = GridSearchCV(
        model_instance,
        param_grid=parameter_grid,
        cv=cross_validation,
        scoring="roc_auc"
    )

    grid_search.fit(X, y)
    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))

    return grid_search
