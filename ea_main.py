# Importing modules

import ea_load
import ea_pca
import ea_classify
import ea_lda
import ea_cluster

path = './data'
output_dir = './output'
#output_dir = ''

options={
        'landscapes': ['CBL', 'CBF', 'Cork'],
         'HUB': 0,
         'OU': 1,
         'CAPA': 1,
         'PLATF': 1,
         'CLASS': 'Settlement'}

df_app = ea_load.load(path, output_dir, options)

ea_lda.fit_ea_lda(df_app, output_dir, options)
ea_pca.reduce_pca(df_app, output_dir, options)
ea_cluster.do_clustering(df_app, output_dir, options)

if options['CLASS'] != '':
    ea_classify.lr_classify(df_app, output_dir, options)
    foo=1

foo=1