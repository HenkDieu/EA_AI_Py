# Importing modules
import pandas as pd
import os
from collections import defaultdict
import csv
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import pickle
import re as re

def old_slow_df_load_ddl(path, abbr_dict, aggr_level):

    included_filename_stems = ['ddl']
    filenames = [fn for fn in os.listdir(path)
                  if any(filename_stem in fn for filename_stem in included_filename_stems)]

    df_schema = pd.DataFrame()

    for filename in filenames:
        # do your stuff
        print('Loading filename = {}'.format(path + '/' + filename))

        index = 0
        with open(path + '/' + filename) as csv_file:
            index += 1
            df = pd.read_csv(csv_file)
            if (aggr_level == 'OWNER'):
                df_owner = df.groupby('OWNER')
            elif (aggr_level == 'TABLE'):
                df_owner = df.groupby(['OWNER', 'TABLE_NAME'])

            for owner_name, group in df_owner:
                tmp_k = list(owner_name)
                tmp_k.append(str(index))
                db_owner_key = '_'.join(tmp_k)

                owner_group = df_owner.get_group(owner_name)

                for row in owner_group.values:
                    for word in (row[1], row[2]):
                        for syll in word.split('_'):
                            if syll in abbr_dict:
                                map_abbr_list = abbr_dict[syll]
                                for map_abbr in map_abbr_list:
                                    if db_owner_key in df_schema.index:
                                        if map_abbr in df_schema.columns:
                                            old_val = df_schema.get_value(db_owner_key, map_abbr)
                                            if pd.isnull(df_schema.loc[db_owner_key, map_abbr]):
                                                df_schema.set_value(db_owner_key, map_abbr, 1)
                                            else:
                                                df_schema.set_value(db_owner_key, map_abbr, old_val+1)
                                        else:
                                            df_schema.set_value(db_owner_key, map_abbr, 1)
                                    else:
                                        df_schema.set_value(db_owner_key, map_abbr, 1)

    return df_schema

def df_load(path, abbr_dict, output_dir, filename_out, options):

    rows_dict_list = []
    rows_dict_index_list = []

    included_filename_stems = ['ddl']
    filenames = [fn for fn in os.listdir(path)
                 if any(filename_stem in fn for filename_stem in included_filename_stems)]

    for filename in filenames:
        # do your stuff
        print('Loading filename = {}'.format(path + '/' + filename))

        with open(path + '/' + filename) as csv_file:
            df = pd.read_csv(csv_file)
            if (options['aggr_level'] == 'OWNER'):
                df_owner = df.groupby('OWNER')
            elif (options['aggr_level'] == 'TABLE'):
                df_owner = df.groupby(['OWNER', 'TABLE_NAME'])

            for owner_name, group in df_owner:
                if (options['aggr_level'] == 'OWNER'):
                    if not re.match("\w*OWNER\w*", owner_name.upper()):
                        continue
                    if re.match("\w*LOG_OWNER\w*|\w*SVCINFR_OWNER\w*|\w*ARCH_OWNER\w*|\w*AQ_OWNER\w*",
                                owner_name.upper()):
                        continue
                else:
                    if not re.match("\w*OWNER\w*", owner_name[0].upper()):
                        continue
                    if re.match("\w*LOG_OWNER\w*|\w*SVCINFR_OWNER\w*|\w*ARCH_OWNER\w*|\w*AQ_OWNER\w*|\w*TI_OWNER\w*|\w*TIPEX_OWNER\w*", owner_name[0].upper()):
                        continue

                tmp_dict = dict()
                owner_group = df_owner.get_group(owner_name)
                # Add table name
                for row in owner_group.values:
                    for word in (row[1], row[2]):
                        for syll in word.split('_'):
                            if syll in abbr_dict:
                                map_abbr_list = abbr_dict[syll]
                                for map_abbr in map_abbr_list:
                                    if map_abbr in tmp_dict:
                                        tmp_dict[map_abbr] += 1
                                    else:
                                        tmp_dict[map_abbr] = 1

                rows_dict_list.append(tmp_dict)
                rows_dict_index_list.append(filename.upper() + '_' + '_'.join(list(owner_name)))

    df_ddl_as_words = pd.DataFrame(rows_dict_list)
    df_ddl_as_words['__ZZ__'] = rows_dict_index_list
    df_ddl_as_words.set_index('__ZZ__', inplace=True)

    pickle.dump(df_ddl_as_words, open(os.path.join(output_dir, filename_out), 'wb'))

    return df_ddl_as_words

def load_abbr(filename):

    abbr_dict = defaultdict()

    with open(filename) as csv_file:
        df_full = pd.read_csv(csv_file)
        df = df_full.query('Include == "Y"')
        df.set_index('Physical Name')
        df_phys_name = df.groupby('Physical Name')
        for physical_name, group in df_phys_name:
            abbr_dict[physical_name] = []
            phys_name_group = df_phys_name.get_group(physical_name)
            abbr_dict[physical_name].append(list([row[2] for row in phys_name_group.values])[0])

    return abbr_dict
