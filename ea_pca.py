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

import ea_decode

def reduce_pca(df_app, output_dir, options):

    options_dscp = ea_decode.decode_options(options)

    df_app_tmp = df_app.copy()
    df_app_tmp.drop('ANNOTATE', axis=1, inplace = True)
    if options['CLASS'] != '':
        df_app_tmp.drop('CLASS', axis=1, inplace=True)

    scaler_list = (StandardScaler(), RobustScaler(), QuantileTransformer(), Normalizer())
    scaler_svd_format = {'StandardScaler': 'r--', 'RobustScaler': 'b--', 'QuantileTransformer': 'g--', 'Normalizer': 'k--'}
    pca = PCA()

    proj_error_dict = dict()

    for scaler in scaler_list:
        scaler_text = 'Preprocessing: ' + scaler.__class__.__name__
        landscape_text = 'Landscapes ' + ' '.join(landscape for landscape in options['landscapes'])
        hub_text ='{}'.format('HUB') if options['HUB'] == 1 else ''

        df_app_tmp2 = scaler.fit_transform(df_app_tmp)

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

        if 0:
            # K-Means on first 2 dimensions
            dim_range = (0, 1)
            nbr_clusters = 5
            kmeans = KMeans(n_clusters=nbr_clusters).fit(ea_new[:,dim_range])

            # Plot
            plt.figure()  # size in inches
            cmap = plt.get_cmap('Set1')
            plt.scatter(x, y, s=area, c=np.array(cmap.colors)[kmeans.labels_], alpha=0.5)
            plt.title(ea_decode.options_title(options) + '\n' + scaler_text + 'KMeans {} clusters'.format(nbr_clusters))
            plt.xlabel('x')
            plt.ylabel('y')

            for x_coord, y_coord, annotation, cluster in zip(x, y, df_app['ANNOTATE'], kmeans.labels_):
                plt.annotate(annotation, xy=(x_coord, y_coord), c=np.array(cmap.colors)[cluster])

            if output_dir == '':
                plt.show()
            else:
                file_out = ea_decode.options_filename(options) + '_' + 'KMeans_{}'.format(nbr_clusters)
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

