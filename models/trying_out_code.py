# -*- coding: utf-8 -*-
"""
project: 

contact: 

link: 

date created: 08.04.2021

@author: Tobias Merk DIYD1
"""


import numpy as np
import pandas as pd
from tqdm import tqdm
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sqlalchemy import create_engine
from collections import Counter

import nltk
nltk.download('stopwords')


def tokenize(text):
    # remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens if tok not in stopwords.words('english')]
    clean_tokens = [lemmatizer.lemmatize(tok, pos='n').lower().strip() for tok in tokens]

    # clean_tokens = []
    # for tok in tokens:
    #     if tok not in stopwords.words():
    #         clean_tok = lemmatizer.lemmatize(tok).lower().strip()
    #         clean_tokens.append(clean_tok)

    return clean_tokens


engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_response_data', engine)


df_cat = df.loc[:, 'related':].mean().sort_values(ascending=False)
df_cat[:5].index
df_cat[:5].values
df_cat[-5:]





# all_tokens = []
# with tqdm(total=len(df), desc='Processing...') as pbar:
#     for token in df['message']:
#         all_tokens += tokenize(token)
#         pbar.update(1)
#
# count_dict = Counter(all_tokens)
# sorted_count_dict = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
# # print(sorted_count_dict[:10])
#
# k = 0
# no_stopword_counter = 0
# words = []
# counts = []
#
# while no_stopword_counter < 5:
#     if sorted_count_dict[k][0] not in stopwords.words('english'):
#         words.append(sorted_count_dict[k][0])
#         counts.append(sorted_count_dict[k][1])
#         # print(sorted_count_dict[k])
#         no_stopword_counter += 1
#     k += 1

# print(sorted_count_dict[0][0])

# tokenize('This is a senctence to test the function!')
#
# stopwords.words('english')
