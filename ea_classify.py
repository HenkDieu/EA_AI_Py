# Importing modules
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split, LeaveOneOut

import logging

from sklearn.base import BaseEstimator

class ClfSwitcher(BaseEstimator):

    def __init__(self, estimator=svm.SVC()):
        self.estimator = estimator

    def get_classes(self):
        return self.estimator.classes_

    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        print('FOO %s\n', self.estimator.__class__)
        return self.estimator.score(X, y)


def classify(df_app, output_dir, options):

    scalers = {
        'Standard': StandardScaler(),
        'Robust': RobustScaler(),
        'QuantileTransform': QuantileTransformer(),
        'Normalizer': Normalizer()
    }

    scaler_params = {
    }

    param_grid = [
        {
            'clf__estimator': [LogisticRegression()],
            'clf__estimator__C': np.logspace(-4, 12, base=2, num=17),
        },
        {
            'clf__estimator': [svm.SVC()],
            'clf__estimator__C': [1, 10, 100, 1000],
            'clf__estimator__kernel': ['linear']
        },
        {
            'clf__estimator': [svm.SVC()],
            'clf__estimator__C': [1, 10, 100, 1000],
            'clf__estimator__gamma': [0.001, 0.0001],
            'clf__estimator__kernel': ['rbf']
        },
        {
            'clf__estimator': [svm.SVC()],
            'clf__estimator__C': [1, 10, 100, 1000],
            'clf__estimator__degree': [2, 3],
            'clf__estimator__kernel': ['poly'],
        }
    ]

    training_param = {
    'training_verbose': 0,
    'test_data_size': 0.3,
    'cross_validation': LeaveOneOut()
    }

    df_app_tmp = df_app.copy()
    df_app_tmp.drop('ANNOTATE', axis=1, inplace = True)
    if options['CLASS'] != '':
        df_app_tmp.drop('CLASS', axis=1, inplace=True)

    for scaler_name, scaler in scalers.items():
        X = df_app_tmp
        Y = df_app['CLASS']
        seed = 7

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=training_param['test_data_size'], random_state=seed)

        pipeline = Pipeline(
            [
                ('scale', scaler),
                ('pca', PCA()),
                #('select', SelectPercentile(chi2), percentile=10),
                ('clf', ClfSwitcher())
            ]
        )

        grid_search = GridSearchCV(pipeline, param_grid = param_grid, scoring = 'f1', verbose=training_param['training_verbose'], cv=training_param['cross_validation'])
        grid_search.fit(X_train, Y_train)
        pred_train = grid_search.predict(X_train)

        pred_test = grid_search.predict(X_test)
        labels = grid_search.best_estimator_._final_estimator.get_classes()
        class_report = metrics.classification_report(Y_test, pred_test, labels=labels)
        logging.info('\nScaler %s - TRAIN Model Test Report: \n%s', scaler_name, class_report)
