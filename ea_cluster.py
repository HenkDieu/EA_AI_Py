import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import ea_decode
from sklearn.neighbors import kneighbors_graph
from sklearn import cluster, datasets, mixture

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from itertools import cycle, islice

def do_clustering(df_app, output_dir, options):

    df_app_tmp = df_app.copy()
    df_app_tmp.drop('ANNOTATE', axis=1, inplace=True)
    if options['CLASS'] != '':
        df_app_tmp.drop('CLASS', axis=1, inplace=True)

    scaler_list = (StandardScaler(), RobustScaler(), QuantileTransformer(), Normalizer())

    for scaler in scaler_list:
        scaler_text = 'Preprocessing: ' + scaler.__class__.__name__

        pca = PCA()

        df_app_tmp2 = scaler.fit_transform(df_app_tmp)
        ea_new = pca.fit_transform(df_app_tmp2)
        X = ea_new
        Y = df_app['CLASS']

        params = {'quantile': .3,
                        'eps': .3,
                        'damping': .9,
                        'preference': -200,
                        'n_neighbors': 10,
                        'n_clusters': 3,
                        'min_samples': 20,
                        'xi': 0.05,
                        'min_cluster_size': 0.1}

        # estimate bandwidth for mean shift
        bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])
        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(
            X, n_neighbors=params['n_neighbors'], include_self=False)
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)

        # ============
        # Create cluster objects
        # ============

        ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
        two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
        ward = cluster.AgglomerativeClustering(
            n_clusters=params['n_clusters'], linkage='ward',
            connectivity=connectivity)
        spectral = cluster.SpectralClustering(
            n_clusters=params['n_clusters'], eigen_solver='arpack',
            affinity="nearest_neighbors")
        dbscan = cluster.DBSCAN(eps=params['eps'])
        optics = cluster.OPTICS(min_samples=params['min_samples'],
                                xi=params['xi'],
                                min_cluster_size=params['min_cluster_size'])
        affinity_propagation = cluster.AffinityPropagation(
            damping=params['damping'], preference=params['preference'])
        average_linkage = cluster.AgglomerativeClustering(
            linkage="average", affinity="cityblock",
            n_clusters=params['n_clusters'], connectivity=connectivity)
        birch = cluster.Birch(n_clusters=params['n_clusters'])
        gmm = mixture.GaussianMixture(
            n_components=params['n_clusters'], covariance_type='full')
        kmeans = cluster.KMeans(
            n_clusters=params['n_clusters'])

        clustering_algorithms = (
            ('MiniBatchKMeans', two_means),
            ('AffinityPropagation', affinity_propagation),
            ('MeanShift', ms),
            ('SpectralClustering', spectral),
            ('Ward', ward),
            ('AgglomerativeClustering', average_linkage),
            ('DBSCAN', dbscan),
            ('OPTICS', optics),
            ('Birch', birch),
            ('GaussianMixture', gmm),
            ('KMeans', kmeans)
        )

        for algo_name, algorithm in clustering_algorithms:
            t0 = time.time()

            # catch warnings related to kneighbors_graph
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="the number of connected components of the " +
                    "connectivity matrix is [0-9]{1,2}" +
                    " > 1. Completing it to avoid stopping the tree early.",
                    category=UserWarning)
                warnings.filterwarnings(
                    "ignore",
                    message="Graph is not fully connected, spectral embedding" +
                    " may not work as expected.",
                    category=UserWarning)
                algorithm.fit(X)

            t1 = time.time()
            if hasattr(algorithm, 'labels_'):
                y_pred = algorithm.labels_.astype(np.int)
            else:
                y_pred = algorithm.predict(X)

            colors = (0, 0, 0)
            area = np.pi * 3

            # Plot
            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                 '#f781bf', '#a65628', '#984ea3',
                                                 '#999999', '#e41a1c', '#dede00']),
                                          int(max(y_pred) + 1))))
            # add black color for outliers (if any)
            colors = np.append(colors, ["#000000"])

            plt.figure()  # size in inches
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])
            plt.title(ea_decode.options_title(options) + '\n' + scaler_text + ' - Clustering: ' + algo_name + ' ({})'.format(params['n_clusters']))
            plt.xlabel('x')
            plt.ylabel('y')

            for x_coord, y_coord, annotation, label in zip(X[:,0], X[:,1], df_app['ANNOTATE'], y_pred):
                plt.annotate(annotation, xy=(x_coord, y_coord), c=colors[label])

            file_out = ea_decode.options_filename(options)  + '_' + scaler.__class__.__name__ + '_cluster' + '_' + algo_name + '_' + '{}'.format(params['n_clusters'])

            if output_dir == '':
                plt.show()
            else:
                plt.savefig(os.path.join(output_dir, file_out))
                plt.close()

            foo=1
