import json
import plotly
import pandas as pd
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
# from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine

from collections import Counter

import nltk
nltk.download('stopwords')


app = Flask(__name__)


def tokenize(text):

    # remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = [lemmatizer.lemmatize(tok, pos='n').lower().strip() for tok in tokens]

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_response_data', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
# @app.route('/')
# @app.route('/index')
# def index():
#     # MAYBE USE THE 10 MOST USED WORD OR SUCH
#
#     # extract data needed for visuals
#     # TODO: Below is an example - modify to extract data for your own visuals
#     genre_counts = df.groupby('genre').count()['message']
#     genre_names = list(genre_counts.index)
#
#     # create visuals
#     # TODO: Below is an example - modify to create your own visuals
#     graphs = [
#         {
#             'data': [
#                 Bar(
#                     x=genre_names,
#                     y=genre_counts
#                 )
#             ],
#
#             'layout': {
#                 'title': 'Distribution of Message Genres',
#                 'yaxis': {
#                     'title': "Count"
#                 },
#                 'xaxis': {
#                     'title': "Genre"
#                 }
#             }
#         }
#     ]
#
#     # encode plotly graphs in JSON
#     ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
#     graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
#
#     # render web page with plotly graphs
#     return render_template('master.html', ids=ids, graphJSON=graphJSON)


@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals

    # 1) genre distribution
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # category distribution
    category_dist = df.loc[:, 'related':].mean().sort_values(ascending=False)

    # 2) most frequent
    category_name_high_freq = category_dist[:5].index
    category_count_high_freq = category_dist[:5].values

    # 3) least frequent
    category_name_least_freq = category_dist[-5:].index
    category_count_least_freq = category_dist[-5:].values



    # # find most frequent words
    # all_tokens = []
    # for token in df['message']:
    #     all_tokens += tokenize(token)
    #
    # # count and sort tokenized words
    # count_dict = Counter(all_tokens)
    # sorted_count_dict = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
    #
    # # find most frequent words that are not stopwords by iterating through the words
    # k = 0
    # no_stopword_counter = 0
    # words = []
    # word_counts = []
    #
    # while no_stopword_counter < 5:
    #     if sorted_count_dict[k][0] not in stopwords.words('english'):
    #         words.append(sorted_count_dict[k][0])
    #         word_counts.append(sorted_count_dict[k][1])
    #         # print(sorted_count_dict[k])
    #         no_stopword_counter += 1
    #     k += 1

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker={'color': 'black'}
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre",
                },
            }
        },

        {
            'data': [
                Bar(
                    x=category_name_high_freq,
                    y=category_count_high_freq,
                    marker={'color': 'red'}
                )
            ],

            'layout': {
                'title': 'Top 5 most frequent categories [in %]',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                },

            }
        },

        {
            'data': [
                Bar(
                    x=category_name_least_freq,
                    y=category_count_least_freq,
                    marker={'color': 'gold'}
                )
            ],

            'layout': {
                'title': 'Top 5 least frequent categories [in %]',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                },
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    # app.run(host='0.0.0.0', port=3001, debug=True)
    app.run()


if __name__ == '__main__':
    main()
