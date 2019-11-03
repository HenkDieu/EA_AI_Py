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
import ddl_load
import ddl_lda
import word_misc
import pickle

def map_df_voca_to_dict_text(df_schema):

    schema_series = defaultdict()
    word_series = defaultdict()

    for (row_idx, row) in df_schema.iterrows():
        sel_col = list(np.where(row>0))
        text_list = []
        for idx in sel_col[0]:
            colname = df_schema.columns[idx]
            text_list.append((colname + ' ') * int(row[idx]))

        schema_series[row_idx] = row_idx
        word_series[row_idx] = ' '.join(text_list)

    df_text = pd.DataFrame({'Schema_Owner': schema_series, 'Words': word_series})
    df_text.set_index('Schema_Owner', inplace=True)

    return df_text

options = {}
input_dir = './abbr'
filename = 'abbreviations.csv'
abbr_dict = ddl_load.load_abbr(os.path.join(input_dir, filename))

input_dir = './data'
output_dir = './output'
filename_stem = 'DDL_LDA'
#options['aggr_level'] = 'TABLE'
options['aggr_level'] = 'OWNER'
df_model_filename = filename_stem + '_' + 'WC' + '_' + 'DATAFRAME.sav'

df_schema = ddl_load.df_load(input_dir, abbr_dict, output_dir, df_model_filename, options)
df_schema = pickle.load(open(os.path.join(output_dir, df_model_filename), 'rb'))

words_sentences = map_df_voca_to_dict_text(df_schema)['Words']

# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(stop_words='english')
# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(words_sentences)
word_misc.plot_most_common_words(count_data, count_vectorizer, 20)
word_misc.makeImage(df_schema.sum(), output_dir, filename_stem)

nbr_topics=5
lda = ddl_lda.fit_ddl_lda(words_sentences, output_dir, filename_stem, nbr_topics)
