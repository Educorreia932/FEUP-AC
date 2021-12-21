import os
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

from datetime import datetime
from dateutil.relativedelta import relativedelta
from scipy.sparse import data

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from IPython.display import Markdown, display


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
        X = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)

    return X, y


def tune_model(
    dataset,
    model_instance,
    parameter_grid,
    columns_to_drop,
    target_column,
    cross_validation=StratifiedKFold(n_splits=5),
    scaler=None,
    feature_selection=False,
    oversample=False
):
    X, y = get_X_y(dataset, columns_to_drop, target_column, scaler)

    instance_parameter_grid = {}

    for parameter_name, parameter_values in parameter_grid.items():
        instance_parameter_grid[f"model__{parameter_name}"] = parameter_values

    parameter_grid = instance_parameter_grid

    steps = []

    if feature_selection:
        rfe = SelectKBest(f_classif, k=10)
        steps.append(('feature_selection', rfe))

    if oversample:
        steps.append(('sampling', SMOTE(n_jobs=-1)))

    steps.append(("model", model_instance))

    estimator = Pipeline(steps=steps)

    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html
    scoring = {"AUC": "roc_auc", "Accuracy": make_scorer(
        accuracy_score), "F1": make_scorer(f1_score)}

    grid_search = GridSearchCV(
        estimator,
        param_grid=parameter_grid,
        cv=cross_validation,
        scoring=scoring,
        refit="AUC"
    )

    grid_search.fit(X, y)

    display(Markdown(f"**Best score:** {grid_search.best_score_}"))
    display(Markdown(f"**Best parameters:** {grid_search.best_params_}"))

    return grid_search

# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html


def plotROC(grid_search_list,
            dataset,
            columns_to_drop,
            target_column,
            scaler=None):

    X, y = get_X_y(dataset, columns_to_drop, target_column, scaler)

    labels = ["No Feature selection/No oversampling", "Feature Selection",
              "Oversampling", "Feature Selection/Oversampling"]

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(1, len(grid_search_list), figsize=(35, 7))
    for j in range(len(grid_search_list)):
        axj = fig.get_axes()[j]
        grid_search = grid_search_list[j]
        for i, (train, test) in enumerate(grid_search.cv.split(X, y)):
            grid_search.best_estimator_.fit(X.iloc[train], y.iloc[train])
            viz = RocCurveDisplay.from_estimator(
                grid_search.best_estimator_,
                X.iloc[test],
                y.iloc[test],
                name="ROC fold {}".format(i),
                alpha=0.3,
                lw=1,
                ax=axj,
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

        axj.plot([0, 1], [0, 1], linestyle="--", lw=2,
                 color="r", label="Chance", alpha=0.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        axj.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        axj.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

        axj.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            title=labels[j],
        )
        
        axj.legend(loc="lower right")

    plt.show()


def plotAlgorithmROC(
    grid_search_list,
    labels,
    dataset,
    columns_to_drop,
    target_column,
    scalers
):
    fig, axs = plt.subplots(1, figsize=(10, 10))

    for i in range(len(grid_search_list)):
        X, y = get_X_y(
            dataset,
            columns_to_drop,
            target_column,
            StandardScaler() if scalers[i] else None
        )

        RocCurveDisplay.from_estimator(grid_search_list[i].best_estimator_, X, y, name=labels[i], ax=axs)

    plt.show()


def confMatrix(models, columns_to_drop, target, dataset, scaler=None):
    titles = [
        "No Oversampling/No Feature Selection",
        "Feature Selection",
        "Oversampling",
        "Feature Selection/Oversampling"
    ]

    fig, axs = plt.subplots(1, 4, figsize=(30, 5))

    X, y = get_X_y(dataset, columns_to_drop, target, scaler)

    for i in range(len(models)):
        cf_matrix = confusion_matrix(y, models[i].predict(X), labels=None, sample_weight=None, normalize=None)
        sb.heatmap(cf_matrix, annot=True, fmt="d", ax=axs[i])
        axs[i].set_title(titles[i])
        axs[i].set_xlabel("Predicted")
        axs[i].set_ylabel("Actual")

    plt.show()
