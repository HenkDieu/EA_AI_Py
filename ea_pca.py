# Importing modules
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
#from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from operator import itemgetter

import ea_decode

def reduce_pca(df_app, output_dir, options, Kfeatures=5):

    options_dscp = ea_decode.decode_options(options)

    df_app_tmp = df_app.copy()
    df_app_tmp.drop('ANNOTATE', axis=1, inplace = True)
    if options['CLASS'] != '':
        df_app_tmp.drop('CLASS', axis=1, inplace=True)

    # Univariate Feature Selection
    X = df_app_tmp
    Y = df_app['CLASS']

    univar_select = SelectKBest(score_func=chi2, k=Kfeatures)
    univar_select.fit(X, Y)
    # summarize scores
    sorted_idx, sorted_val = zip(*sorted([(i, e) for i, e in enumerate(univar_select.scores_)], key=itemgetter(1), reverse=True))
    print('Univariate feature selection')
    for idx, valn in zip(sorted_idx, sorted_val):
        print('\tFeature {}: {}'.format(df_app_tmp.columns.values[idx], valn))

    scaler_list = (StandardScaler(), RobustScaler(), QuantileTransformer(output_distribution='normal'), Normalizer())
    scaler_svd_format = {'StandardScaler': 'r--', 'RobustScaler': 'b--', 'QuantileTransformer': 'g--', 'Normalizer': 'k--'}

    proj_error_dict = dict()

    for scaler in scaler_list:
        scaler_text = 'Preprocessing: ' + scaler.__class__.__name__
        landscape_text = 'Landscapes ' + ' '.join(landscape for landscape in options['landscapes'])
        hub_text ='{}'.format('HUB') if options['HUB'] == 1 else ''

        df_app_tmp2 = scaler.fit_transform(df_app_tmp)

        X = df_app_tmp2
        Y = df_app['CLASS']

        # ExtraTrees Feature Selection
        extra_select = ExtraTreesClassifier(n_estimators=Kfeatures)
        extra_select.fit(X, Y)
        print('ExtraTrees Feature Classification')
        sorted_idx, sorted_val = zip(*sorted([(i, e) for i, e in enumerate(extra_select.feature_importances_)], key=itemgetter(1), reverse=True))
        for idx, valn in zip(sorted_idx, sorted_val):
            print('\tFeature {}: {}'.format(df_app_tmp.columns.values[idx], valn))

        # Recursive Feature Extraction
        lr_select = LogisticRegression(solver='lbfgs')
        rfe = RFE(lr_select, Kfeatures)
        rfe.fit(X, Y)
        print("Num Features: %d" % rfe.n_features_)
        for i in [i for i, x in enumerate(rfe.support_) if x]:
            print("\tFeature: %s" % df_app_tmp.columns.values[i])
        sorted_idx, sorted_val = zip(*sorted([(i, e) for i, e in enumerate(rfe.ranking_)], key=itemgetter(1)))
        for idx, valn in zip(sorted_idx, sorted_val):
            print("\tFeature %s: ranking %d" % (df_app_tmp.columns.values[idx], valn))

        # Principal Component Analysis
        pca = PCA()
        ea_new = pca.fit_transform(df_app_tmp2)
        #print('{}\n\n'.format(pca.explained_variance_ratio_))

        n_dim = len(pca.explained_variance_ratio_)
        tril_ones = np.tril(np.ones([n_dim, n_dim]), 0)
        residual_error = 1-np.dot(tril_ones, pca.explained_variance_ratio_)

        proj_error_dict[scaler.__class__.__name__]=residual_error

        index = min(np.where(residual_error <= 0.2)[0])
        res_error_2 = residual_error[1]
        print('{} {} {} \nSVD Error <= 20%: after {} dimensions -  Error after 2 dimensions: {}'.format(landscape_text, scaler_text, options_dscp, index, res_error_2))

        file_out = ea_decode.options_filename(options) + '_' + scaler.__class__.__name__ + '_landscape'

        x = ea_new[:,0]
        y = ea_new[:,1]
        colors = (0, 0, 0)
        area = np.pi * 3

        # Plot
        plt.figure()  # size in inches
        plt.scatter(x, y, s=area, c="blue", alpha=0.5)
        plt.title(ea_decode.options_title(options) + '\n' + scaler_text)
        plt.xlabel('x')
        plt.ylabel('y')

        for x_coord, y_coord, annotation in zip(x, y, df_app['ANNOTATE']):
            plt.annotate(annotation, xy=(x_coord, y_coord), c="blue")

        if output_dir == '':
            plt.show()
        else:
            plt.savefig(os.path.join(output_dir, file_out))
            plt.close()


    file_out = ea_decode.options_filename(options) + '_projerror'
    plt.figure()  # size in inches
    for scaler_name in proj_error_dict:
        residual_error_t = proj_error_dict[scaler_name]
        format = scaler_svd_format[scaler_name]
        plt.plot(range(len(pca.explained_variance_ratio_)), residual_error_t, format, label=scaler_name)
    plt.title(ea_decode.options_title(options) + '\n' + 'SVD Projection Error')
    plt.axis([0, len(pca.explained_variance_ratio_), 0, 1])
    plt.legend(loc='best', shadow=True, fontsize='x-large')

    if output_dir == '':
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, file_out))
        plt.close()

