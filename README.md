# Disaster Response Pipeline Project

### About The Project
Repository contains code (and data) to categorize disaster messages using a supervised machine learning model.

### Project Organisation
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # input data
|- disaster_messages.csv  # input data
|- process_data.py      # code to preprocess data
|- InsertDatabaseName.db   # database to save cleaned data to

- models
|- train_classifier.py  # code to train classifier and export model
|- classifier.pkl  # saved model 
```

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
