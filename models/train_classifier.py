import sys
import pickle
import pandas as pd
import re

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

from datetime import datetime

import nltk
nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    """
    load data from database and select predictors and responses

    Parameters
    ----------
    database_filepath : path where to find database file

    Returns
    -------
    X : predictors
    Y : response
    categories : response categories

    """

    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_response_data', con=engine)
    df = df.loc[:500, :]

    # select response categories, predictors and responses
    categories = df.loc[:, 'related':].columns
    X = df['message'].values
    Y = df[categories].values

    return X, Y, categories


def tokenize(text):
    """
    tokenizes text and sets to lower case

    Parameters
    ----------
    text : text to be tokenized

    Returns
    -------
    clean_tokens : tokenized text

    """

    # remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    # tokenize text
    tokens = word_tokenize(text)

    # remove stopwords --> issue with parallel jobs!!!
    # tokens = [tok for tok in tokens if tok not in stopwords.words('english')]

    # lemmatize words and set to lower case
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok, pos='n').lower().strip() for tok in tokens]

    return clean_tokens


def build_model():
    """
    machine learning pipeline to train and optimize a MultiOutputClassifier using GridSearchCV

    Returns
    -------

    cv : GridSearchCV object

    """

    # pipeline RandomForest
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])

    # parameters for grid search to optimize model --> choose
    # more parameters makes calculation more expensive
    # may run a few hours depending on the chosen parameters
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        'clf__estimator__max_depth': (None, 10, 20),
        'clf__estimator__n_estimators': [100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }

    # pipeline KNN
    # pipeline = Pipeline([
    #     ('vect', CountVectorizer(tokenizer=tokenize)),
    #     ('tfidf', TfidfTransformer()),
    #     ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    # ])
    #
    # parameters = {
    #     'vect__ngram_range': ((1, 1), (1, 2)),
    #     'vect__max_df': (0.5, 0.75, 1.0),
    #     'vect__max_features': (None, 1000, 5000, 10000),
    #     # 'tfidf__use_idf': (True, False),
    #     # 'clf__estimator__n_neighbors': [5, 11],
    #     # 'clf__estimator__leaf_size': [20, 30, 50]
    # }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)

    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """
    evaluate model using sklearn's classification_report

    Parameters
    ----------
    model : NLP model
    X_test : test predictors
    y_test : test responses
    category_names : response categories

    Returns
    -------

    """

    df_out = pd.DataFrame(columns=['category', 'recall', 'precision', 'f1-score'])

    y_pred = model.predict(X_test)

    for ind, cat in enumerate(category_names):
        report_dict = classification_report(y_test[:, ind], y_pred[:, ind], output_dict=True)

        df_out.loc[ind, 'category'] = cat
        df_out.loc[ind, 'recall'] = report_dict['weighted avg']['recall']
        df_out.loc[ind, 'precision'] = report_dict['weighted avg']['precision']
        df_out.loc[ind, 'f1-score'] = report_dict['weighted avg']['f1-score']

    df_out.to_excel('model_evaluation.xlsx', index=False)
    print(df_out)


def save_model(model, model_filepath):
    """
    export model to pickle file

    Parameters
    ----------
    model : trained model
    model_filepath : file path where to save the model

    Returns
    -------

    """

    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    """
    main function that calls all other functions

    Returns
    -------

    """

    start_time = datetime.now()

    # make sure of correct amount of arguments
    if len(sys.argv) == 3:

        # read arguments
        database_filepath, model_filepath = sys.argv[1:]

        # load data
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        # split data into training/test sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

        # build model
        print('Building model...')
        model = build_model()

        # train model
        print('Training model...')
        model.fit(X_train, Y_train)

        # evaluate model to check fit
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        # print best parameters
        print(model.best_params_)

        # save model
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))


if __name__ == '__main__':
    main()
