from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.naive_bayes import GaussianNB
import pickle
import re
import pandas as pd
from textblob import TextBlob

model = pickle.load(open('Pickle_svm_sameday_new1.pkl', 'rb'))
#!pip install joblib
#from sklearn.externals import joblib


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('HOme.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Accepting the user input
        # Tweets=user input
        Tweets = request.form['Tweet/Headline']
        df_copy = str(Tweets)
        #Cleaning and preprocessing
        df_copy = df_copy.lower()
        # replace username-tags
        df_copy = re.sub(r'@[A-Z0-9a-z_:]+', '', df_copy)
        df_copy = re.sub(r'^[RT]+', '', df_copy)  # replace RT-tags
        df_copy = re.sub('https?://[A-Za-z0-9./]+',
                         '', df_copy)  # replace URLs
        df_copy = re.sub("[^a-zA-Z]", " ", df_copy)  # replace hashtags
        # Create textblob objects of the tweet
        sentiment_objects = TextBlob(df_copy)
        tweet_polarity = sentiment_objects.sentiment.polarity
        result = [tweet_polarity]
        if tweet_polarity < 0:
            NUM_NEG = 1
        else:
            NUM_NEG = 0
        result.append(NUM_NEG)
        dict = {'polarity': [result[0]], 'NUM_NEG': [result[1]]}
        #dict = {'polarity':[result[0]]}
        df_test = pd.DataFrame(dict)
        # scaling and normalization of the test set
        sc = StandardScaler()
        data = pd.read_csv('labelled_dataset.csv')
        data_1 = data[['lab_sameday', 'Polarity', 'NUM_NEG']]
        X = data_1[['Polarity', 'NUM_NEG']]
        y = data_1['lab_sameday'].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0)
        X_train = sc.fit_transform(X_train)
        df_test = sc.transform(df_test)
        # normalize the data attributes
        X_train = preprocessing.normalize(X_train)
        df_test = preprocessing.normalize(df_test)
        prediction = model.predict(df_test)
        #prediction = model.predict(X_test[0])
        print(prediction[0])
        if prediction[0] == 1:
            return render_template('HOme.html', prediction_text='This Headline/Tweet may increase market volatility')
        else:
            return render_template('HOme.html', prediction_text='This Headline/Tweet may decrease market volatility')


if __name__ == "__main__":
    app.run(port=5000, debug=True)
