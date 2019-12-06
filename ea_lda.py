# Importing modules
import pandas as pd
import os
import re
from collections import defaultdict
import csv
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from pyLDAvis import sklearn as sklearn_lda
import pyLDAvis
import pickle
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import ea_decode

def makeImage(text_dict, output_dir, filename):

    x, y = np.ogrid[:300, :300]
    mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
    mask = 255 * mask.astype(int)

    wc = WordCloud(background_color="white", max_words=1000, mask=mask)
    # generate word cloud
    wc.generate_from_frequencies(text_dict)

    # show
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")

    if output_dir == '':
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

def transform_to_nlp(df_app):

    flatten_text_dict = defaultdict()
    for index, row in df_app.iterrows():
        flatten_text = ''
        for colname in df_app.columns:
            if (row[colname] > 0):
                if index in flatten_text_dict:
                    flatten_text_dict[index] = flatten_text_dict[index] + (' ' + re.sub('[\s|\-|\/]+', '_', colname))*int(row[colname])
                else:
                    flatten_text_dict[index] = (' ' + re.sub('[\s|\-|\/]+', '_', colname)) * int(row[colname])

    df_dict = pd.DataFrame.from_dict(flatten_text_dict, orient='index')
    df_dict.set_axis(['Words'], axis=1, inplace=True)
    df_tmp = df_app.join(df_dict, how='inner', rsuffix='_IN').fillna(0)

    return df_tmp

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

def fit_ea_lda(df_app, output_dir, options):
    # Helper function
    def print_topics(model, count_vectorizer, n_top_words):
        words = count_vectorizer.get_feature_names()
        for topic_idx, topic in enumerate(model.components_):
            print("\nTopic #%d:" % topic_idx)
            print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

    df_app_tmp = df_app.copy()
    df_app_tmp.drop('ANNOTATE', axis=1, inplace=True)
    if options['CLASS'] != '':
        df_app_tmp.drop('CLASS', axis=1, inplace=True)

    df_app_words = transform_to_nlp(df_app_tmp)
    makeImage(df_app_tmp.sum(), output_dir, ea_decode.options_filename(options) + '_' + 'WC')
    # Initialise the count vectorizer with the English stop words
    count_vectorizer = CountVectorizer(stop_words='english')
    # Fit and transform the processed titles
    count_data = count_vectorizer.fit_transform(df_app_words['Words'])

    #plot_most_common_words(count_data, count_vectorizer, 15)

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

    file_out = ea_decode.options_filename(options) + '_' + 'LDA_VIS' + '_' + str(number_topics)
    LDAvis_data_filepath = os.path.join(output_dir, file_out)
    # # this is a bit time consuming - make the if statement True
    # # if you want to execute visualization prep yourself
    if 1 == 1:
        LDAvis_prepared = sklearn_lda.prepare(lda, count_data, count_vectorizer)
    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)

    # load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, os.path.join(output_dir, file_out + '.html'))
