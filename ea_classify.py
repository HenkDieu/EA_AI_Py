# Importing modules
import os
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import metrics, feature_selection
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split, LeaveOneOut

import logging
import ea_decode

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
        return self.estimator.score(X, y)


def classify(df_app, output_dir, options):

    scalers = {
        'Standard': StandardScaler(),
        'Robust': RobustScaler(),
        'Normalizer': Normalizer()
    }

    K = [3]
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
            'clf__estimator__gamma': [0.1, 0.01, 0.001, 0.0001],
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
         #       ('feat', feature_selection.SelectKBest()),
                ('clf', ClfSwitcher())
            ]
        )

        grid_search = GridSearchCV(pipeline, param_grid = param_grid, scoring = 'f1', verbose=training_param['training_verbose'], cv=training_param['cross_validation'])
        grid_search.fit(X_train, Y_train)
        pred_train = grid_search.predict(X_train)

        pred_test = grid_search.predict(X_test)
        labels = grid_search.best_estimator_._final_estimator.get_classes()
        class_report = metrics.classification_report(Y_test, pred_test, labels=labels)

        best_clf = grid_search.best_params_['clf__estimator']

        #best_feat_str = ' - ' .join(X.columns[grid_search.best_estimator_.named_steps['feat'].get_support(indices=True)])
        #logging.info('\nScaler %s - TRAIN Model Test Report: \n%s\nSelected features: %s', scaler_name, class_report,best_feat_str)

        logging.info('\nScaler %s - Best classifier %s - TRAIN Model Test Report: \n%s', scaler_name, best_clf.__class__.__name__, class_report)

        # Dimensionality reduction and then apply the retained best classifier and hyperparameters
        # to re-train on reduced 2-dimensional dataset
        # Don't think this makes much sense but as a way of illustrating

        pca = PCA()
        X_scaled = scaler.fit_transform(X)
        #X_pca = pca.fit_transform(X_scaled.to_numpy()[:, grid_search.best_estimator_.named_steps['feat'].get_support(indices=True)])
        X_pca = pca.fit_transform(X_scaled)

        best_clf.fit(X_pca[:,[0,1]], Y)

        h = .02  # step size in the mesh

        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # title for the plot
        plt.figure()
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title(ea_decode.options_title(options) + '\n' + 'Scaler: ' + scaler_name + '\n' + 'Classifier: ' + best_clf.__class__.__name__)

        prediction = best_clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        prediction = prediction.reshape(xx.shape)
        plt.contourf(xx, yy, prediction, cmap=plt.cm.coolwarm, alpha=0.8)

        # Plot also the training points
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y, cmap=plt.cm.coolwarm)
        colors = ('b', 'r')
        for x_coord, y_coord, annotation, label in zip(X_pca[:, 0], X_pca[:, 1], df_app['ANNOTATE'], Y):
            plt.annotate(annotation, xy=(x_coord, y_coord), c=colors[label])

        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.xticks(())
        plt.yticks(())

        file_out = ea_decode.options_filename(options) + '_' + scaler_name + '_' + best_clf.__class__.__name__

        if output_dir == '':
            plt.show()
        else:
            plt.savefig(os.path.join(output_dir, file_out))
            plt.close()

