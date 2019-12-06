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
from pyLDAvis import sklearn as sklearn_lda
import pyLDAvis
import pickle
import multidict as multidict

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

# Read data into schemas

def makeImage(text_dict, output_dir, filename_stem):

    x, y = np.ogrid[:500, :500]
    mask = (x - 250) ** 2 + (y - 250) ** 2 > 230 ** 2
    mask = 255 * mask.astype(int)

    wc = WordCloud(background_color="white", max_words=1000, mask=mask)
    # generate word cloud
    wc.generate_from_frequencies(text_dict)

    # show
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()

    file_out = filename_stem + '_' + 'WC_IMG'
    if output_dir == '':
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, file_out))
        plt.close()


def df_load_ddl(path, abbr_dict, output_dir, filename_out, options):

    rows_dict_list = []
    rows_dict_index_list = []

    included_filename_stems = ['ddl']
    filenames = [fn for fn in os.listdir(path)
                 if any(filename_stem in fn for filename_stem in included_filename_stems)]

    for filename in filenames:
        # do your stuff
        print('Loading filename = {}'.format(path + '/' + filename))

        with open(path + '/' + filename) as csv_file:
            schema_voc_dict = dict()
            df = pd.read_csv(csv_file)
            if (options['aggr_level'] == 'OWNER'):
                df_owner = df.groupby('OWNER')
            elif (options['aggr_level'] == 'TABLE'):
                df_owner = df.groupby(['OWNER', 'TABLE_NAME'])

            for owner_name, group in df_owner:
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
            #abbr_dict[physical_name].extend(list(set([row[2] for row in phys_name_group.values])))
            abbr_dict[physical_name].append(list([row[2] for row in phys_name_group.values])[0])

    return abbr_dict

# Helper function
def plot_most_common_words(count_data, count_vectorizer, top_nbr):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts += t.toarray()[0]

    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:top_nbr]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words))

    plt.figure(2, figsize=(15, 15 / 1.6180))
    plt.subplot(title='{} {}'.format(top_nbr,' most common words'))
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90)
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()

# Helper function
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


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
        # word_series[index] = ' '.join(list(set(tmp_list)))
        word_series[row_idx] = ' '.join(text_list)

    df_text = pd.DataFrame({'Schema_Owner': schema_series, 'Words': word_series})
    df_text.set_index('Schema_Owner', inplace=True)

    return df_text


options = {}
path = './abbr'
filename = 'abbreviations.csv'
abbr_dict = load_abbr(path + '/' + filename)

path = './data'
output_dir = './output'
filename_stem = 'DDL_LDA'
options['aggr_level'] = 'TABLE'
df_model_filename = filename_stem + '_' + 'WC' + '_' + 'DATAFRAME.sav'

#df_schema = df_load_ddl(path, abbr_dict, output_dir, df_model_filename, options)
df_schema = pickle.load(open(os.path.join(output_dir, df_model_filename), 'rb'))
count_data = df_schema.sum()

makeImage(count_data, './output', filename_stem)
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(stop_words='english')

# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(map_df_voca_to_dict_text(df_schema)['Words'])
# Visualise the 10 most common words
plot_most_common_words(count_data, count_vectorizer, 20)

# Load the LDA model from sk-learn
from sklearn.decomposition import LatentDirichletAllocation as LDA


# Tweak the two parameters below
number_topics = 10
number_words = 20
# Create and fit the LDA model
lda = LDA(n_components=number_topics, n_jobs=-1)
lda.fit(count_data)
# Print the topics found by the LDA model
print("Topics found via LDA:")
print_topics(lda, count_vectorizer, number_words)

LDAvis_data_filepath = os.path.join(output_dir, filename_stem + '_' + 'lda_vis_prepared_' + str(number_topics))
# # this is a bit time consuming - make the if statement True
# # if you want to execute visualization prep yourself
if 1 == 1:
    LDAvis_prepared = sklearn_lda.prepare(lda, count_data, count_vectorizer)
with open(LDAvis_data_filepath, 'wb') as f:
    pickle.dump(LDAvis_prepared, f)

# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)
pyLDAvis.save_html(LDAvis_prepared, os.path.join(output_dir, filename_stem + '_' + 'lda_vis_prepared_' + str(number_topics) + '.html'))

foo=1