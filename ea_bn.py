import os
import pickle

import matplotlib.pyplot as plt
from pomegranate import BayesianNetwork
import seaborn, time

import ea_decode
seaborn.set_style('whitegrid')

import logging

def build_bn(df_app, output_dir, options):

    df_app_tmp = df_app.copy()
    df_app_tmp.drop('ANNOTATE', axis=1, inplace = True)
    if options['CLASS'] != '':
        df_app_tmp.drop('CLASS', axis=1, inplace=True)

    X = df_app_tmp
    model = BayesianNetwork.from_samples(X, algorithm='chow-liu')

    print("\nModel Structure:\n")
    print(model.structure)
    for idx, parent in enumerate(model.structure):
        if len(parent) == 0:
            print('Singleton: {}'.format(df_app.columns[idx]))
        elif len(parent) == 1:
            print('Parent: {} - Child: {}'.format(df_app.columns[parent[0]], df_app.columns[idx]))

    file_out = ea_decode.options_filename(options) + '_' + 'bn_graph'

    plt.figure(figsize=(9, 7))
    model.plot()

    if output_dir == '':
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, file_out))
        plt.close()

    file_out = ea_decode.options_filename(options) + '_bn.mdl'
    model_file = os.path.join(output_dir, file_out)

    with open(model_file, 'wb') as f:
        pickle.dump(model_file, f)

    logging.info('\n%s: Loglikelihood: %.2f\n', 'BN', model.log_probability(X).sum())

    #print(model.predict_proba({'0': '3'}))
