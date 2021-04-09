import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    load data from both csv files and merges them on their id

    Parameters
    ----------
    messages_filepath : messages.csv file path
    categories_filepath : categories.csv file path

    Returns
    -------
    df : df containing the merged data from both input files
    categories : response categories

    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    print(messages.shape)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    print(categories.shape)
    categories.head()

    # merge datasets
    df = messages.merge(categories, on='id', how='left')
    print(df.shape)
    df.head()

    return df, categories


def clean_data(df, categories):
    """
    clean data by converting the "categories" column into several columns containing only the
    important information and finally remove duplicates!

    Parameters
    ----------
    df : input data
    categories : response categories

    Returns
    -------

    df : cleaned df

    """
    # create a dataframe of the 36 individual category columns
    categories = categories['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.loc[0, :]

    # use this row to extract a list of new column names for categories.
    category_colnames = row.str.split('-', expand=True)[0]

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.replace('{}-'.format(column), "")

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df = df.drop(columns=['categories'])

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], sort=False, axis=1, join='inner')

    # related contains more 3 categories --> should be binary --> remove rows
    df = df[df['related'] != 2]

    # column "child alone" only contains 0 --> column not relevant --> remove
    df = df.drop(columns=['child_alone'])

    # drop duplicates
    df = df.drop_duplicates()
    print(df.shape)

    # category values should be binary
    # for col in df.loc[:, 'related':]:
    #     print(col)
    #     print(df[col].unique())

    return df


def save_data(df, database_filename):
    """
    export cleaned data to sqlite database

    Parameters
    ----------
    df : cleaned data
    database_filename : (output) database file

    Returns
    -------

    """

    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_response_data', engine, index=False)
    df.to_excel('disaster_response_data.xlsx', index=False)


def main():
    """
    main function. Calls all other functions.

    Returns
    -------

    """

    # make sure there are enough arguments
    if len(sys.argv) == 4:

        # read arguments
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        # load data
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df, categories = load_data(messages_filepath, categories_filepath)

        # clean data
        print('Cleaning data...')
        df = clean_data(df, categories)

        # save data
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv DisasterResponse.db')


if __name__ == '__main__':
    main()
