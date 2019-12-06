# Importing modules

import logging

import ea_load
import ea_pca
import ea_classify
import ea_lda
import ea_cluster
import ea_decode
import ea_bn

if __name__ == '__main__':

    path = './data'
    output_dir = './output'
    # output_dir = ''

    options = {
        'landscapes': ['CBL', 'CBF', 'Cork'],
        'HUB': 0,
        'OU': 1,
        'CAPA': 1,
        'PLATF': 1,
        'CLASS': 'Settlement'}

    logging.basicConfig(filename=output_dir + '/' + 'ea_main.log',
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%d/%b/%Y %H:%M:%S')

    logging.info(ea_decode.decode_options(options))
    df_app = ea_load.load(path, output_dir, options)

    ea_lda.fit_ea_lda(df_app, output_dir, options)
    ea_pca.reduce_pca(df_app, output_dir, options)
    ea_cluster.do_clustering(df_app, output_dir, options)
    ea_bn.build_bn(df_app, output_dir, options)

    if options['CLASS'] != '':
        ea_classify.classify(df_app, output_dir, options)
        foo=1