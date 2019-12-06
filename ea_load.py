# Importing modules
import pandas as pd
import os
import re
from collections import defaultdict
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing.label import LabelBinarizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ea_decode

import pickle

# Read data into schemas

def merge_appl(df, to_app_name, from_app_name):

    dummy_key = 'DUMMY'
    to_app_name_IN = to_app_name + '_IN'
    to_app_name_OUT = to_app_name + '_OUT'
    from_app_name_IN = from_app_name + '_IN'
    from_app_name_OUT = from_app_name + '_OUT'

    df[df.index == to_app_name] = df[df.index == to_app_name].values + df[df.index == from_app_name].values

    df['DUMMY_IN'] = df[from_app_name_IN] + df[to_app_name_IN]
    df['DUMMY_OUT'] = df[from_app_name_OUT] + df[to_app_name_OUT]
    df[to_app_name_IN] = df['DUMMY_IN']
    df[to_app_name_OUT] = df['DUMMY_OUT']
    df.drop('DUMMY_IN', axis=1, inplace=True)
    df.drop('DUMMY_OUT', axis=1, inplace=True)
    df.loc[to_app_name, to_app_name_IN] = 0
    df.loc[to_app_name, to_app_name_OUT] = 0

    df.drop(from_app_name_IN, axis=1, inplace=True)
    df.drop(from_app_name_OUT, axis=1, inplace=True)
    df.drop(from_app_name, axis=0, inplace=True)

def drop_appl(df, app_name):

    app_name_IN = app_name + '_IN'
    app_name_OUT = app_name + '_OUT'

    df.drop(app_name_IN, axis=1, inplace=True)
    df.drop(app_name_OUT, axis=1, inplace=True)
    df.drop(app_name, axis=0, inplace=True)

def load(path, output_dir, options):

    filename = 'app_dbag.csv'
    print('Loading filename = {}'.format(path + '/' + filename))
    app_dict = defaultdict()
    with open(path + '/' + filename) as csv_file:
        df = pd.read_csv(csv_file)
        df_app = df.query("Landscape in ['" + "','".join(options['landscapes']) + "']")
        df_app_details = df_app.copy()

    df_app_details = df_app.set_index('Application')

    df_app = df_app.set_index('Application')
    for colname in ['ANNOTATE', 'AID', 'Product', 'Landscape']:
        df_app.drop(colname, axis=1, inplace=True)

    filename = 'app_coupling.csv'
    print('Loading filename = {}'.format(path + '/' + filename))
    with open(path + '/' + filename) as csv_file:
        df = pd.read_csv(csv_file)
        df['Count'] = 1

        df_tmp = df.pivot_table(index='Sender Application', columns='Receiver Application', values='Count',
                            fill_value=0, aggfunc=np.sum)
        df_app2 = df_app.join(df_tmp.T, how='left').T
        for colname in df_app2.columns:
            df_app2.rename({colname: colname + '_IN'}, axis=1, inplace=True)
        df_app = df_app.join(df_app2, how='left', rsuffix='_IN').fillna(0)

        df_tmp = df.pivot_table(index='Receiver Application', columns='Sender Application', values='Count',
                            fill_value=0, aggfunc=np.sum)
        df_app2 = df_app.join(df_tmp.T, how='left').T
        for colname in df_app2.columns:
            df_app2.rename({colname: colname + '_OUT'}, axis=1, inplace=True)
        df_app = df_app.join(df_app2, how='left', rsuffix='_OUT').fillna(0)

    # Merge HUB and CreationDirect
    if 'CBL' in options['landscapes']:
        merge_appl(df_app, 'HUB', 'CreationDirect')
        if options['HUB'] != 1:
            drop_appl(df_app, 'HUB')

    if options['CAPA'] == 1:
        filename = 'app_capa.csv'
        print('Loading filename = {}'.format(path + '/' + filename))
        with open(path + '/' + filename) as csv_file:
            df = pd.read_csv(csv_file)
            df['Count'] = 1
            df_tmp = df.pivot_table(index='Application', columns='Capability', values='Count', fill_value=0)
            for colname in df_tmp.columns:
                df_tmp.rename({colname: colname + '_CAPA'}, axis=1, inplace=True)
            df_app = df_app.join(df_tmp, how='left').fillna(0)

    if options['OU'] == 1:
        filename = 'app_ou_c.csv'
        print('Loading filename = {}'.format(path + '/' + filename))
        with open(path + '/' + filename) as csv_file:
            lb = LabelBinarizer()
            df = pd.read_csv(csv_file)
            df = df.set_index('Application')
            df_tmp = lb.fit_transform(df['Responsible'])
            if len(lb.classes_) > 2:
                df_tmp = pd.DataFrame(df_tmp, columns=lb.classes_, index=df.index)
            else:
                df_tmp = pd.DataFrame(df_tmp, columns='OU', index=df.index)
            df_app = df_app.join(df_tmp, how='left').fillna(0)

    if options['PLATF'] == 1:
        filename = 'app_platf_c.csv'
        print('Loading filename = {}'.format(path + '/' + filename))
        with open(path + '/' + filename) as csv_file:
            lb = LabelBinarizer()
            df = pd.read_csv(csv_file)
            df = df.set_index('Application')
            df_tmp = lb.fit_transform(df['Platform'])
            if len(lb.classes_) > 2:
                df_tmp = pd.DataFrame(df_tmp, columns=lb.classes_, index=df.index)
            else:
                df_tmp = pd.DataFrame(df_tmp, columns=['PLATFORM'], index=df.index)
            df_app = df_app.join(df_tmp, how='left').fillna(0)

    if options['CLASS'] != '':
        lb = LabelBinarizer()
        df_tmp = lb.fit_transform(df_app_details['Product'])
        if len(lb.classes_) > 2:
            df_tmp2 = pd.DataFrame(df_tmp, columns=lb.classes_, index=df_app_details.index)
            df_tmp2 = df_tmp2[[options['CLASS']]]
            df_tmp2 = df_tmp2.rename(columns={options['CLASS']:'CLASS'})
            df_app = df_app.join(df_tmp2, how='left').fillna(0)
        else:
            df_tmp2 = pd.DataFrame(df_tmp, columns=['CLASS'], index=df_app_details.index)
            class_idx = -1
            for idx, label in enumerate(lb.classes_):
                if label == options['CLASS']:
                    class_idx = idx
            df_tmp2 = (df_tmp2[['CLASS']] == class_idx).astype(int)
            df_app = df_app.join(df_tmp2, how='left').fillna(0)

    df_app = df_app.join(df_app_details['ANNOTATE'], how='left').fillna(0)

    # save the DataFrame to disk

    file_out =  ea_decode.options_filename(options) + '_DATAFRAME.sav'
    pickle.dump(df_app, open(os.path.join(output_dir, file_out), 'wb'))

    return df_app

