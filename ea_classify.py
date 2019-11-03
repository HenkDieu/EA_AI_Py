# Importing modules
import os
import re
from collections import defaultdict
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
import pickle

import ea_decode

def lr_classify(df_app, output_dir, options):

    df_app_tmp = df_app.copy()
    df_app_tmp.drop('ANNOTATE', axis=1, inplace = True)
    if options['CLASS'] != '':
        df_app_tmp.drop('CLASS', axis=1, inplace=True)

    scaler_list = (StandardScaler(), RobustScaler(), QuantileTransformer(), Normalizer())

    pca = PCA()

    for scaler in scaler_list:
        df_app_tmp2 = scaler.fit_transform(df_app_tmp)

        ea_new = pca.fit_transform(df_app_tmp2)

        X = ea_new

        Y = df_app['CLASS']
        test_size = 0.33
        seed = 7
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
        # Fit the model on training set
        model = LogisticRegression()
        model.fit(X_train, Y_train)
        file_out = ea_decode.options_filename(options) + '_' + scaler.__class__.__name__ + '_' + 'LR'

        # save the model to disk
        pickle.dump(model, open(os.path.join(output_dir, file_out), 'wb'))

        # load the model from disk
        loaded_model = pickle.load(open(os.path.join(output_dir, file_out), 'rb'))
        result = loaded_model.score(X_test, Y_test)
        print(ea_decode.options_title(options) + ' - ' + 'Preprocessing: ' + scaler.__class__.__name__ + ' - ' + 'Class: ' + options['CLASS'] + ' - Logistic Regression score: {}'.format(result))
