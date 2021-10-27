#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import re
import pandas as pd
from textblob import TextBlob
model = pickle.load(open('Pickle_SVM_sameday_stock3.pkl', 'rb'))
#!pip install joblib
#from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('HOME.html')
@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
    #Accepting the user inputs tweets/headline,Open,High,low.
    #Do normalization on all data
       Open=request.form['Open']
       Low=request.form['Low']
       High=request.form['High']
       Open=float(Open)
       High=float(High)
       Low=float(Low)
       #processing of twitter data
       Tweets=request.form['Tweet/Headline']
       df_copy=Tweets
       #Cleaning and preprocessing
       df_copy=df_copy.lower()
       df_copy=re.sub(r'@[A-Z0-9a-z_:]+','',df_copy)#replace username-tags
       df_copy=re.sub(r'^[RT]+','',df_copy)#replace RT-tags
       df_copy = re.sub('https?://[A-Za-z0-9./]+','',df_copy)#replace URLs
       df_copy=re.sub("[^a-zA-Z]", " ",df_copy)#replace hashtags
       # Create textblob objects of the tweet
       sentiment_objects = TextBlob(df_copy)
       tweet_polarity=sentiment_objects.sentiment.polarity
       result=[tweet_polarity]
       dict={'Open':[Open],'High':[High],'Low':[Low],'Polarity':[result[0]]} 
       df_test= pd.DataFrame(dict)
       data = pd.read_csv('labelled_dataset_full.csv')
       data_1=data[['Low','Open','High','lab_sameday','Polarity']]
       data_1.dropna(inplace=True)
       X=data_1[['Open','High','Low','Polarity']]
       y=data_1['lab_sameday'].values
       X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)
       # normalize the data attributes
       X_train= preprocessing.normalize(X_train)
       X_test= preprocessing.normalize(X_test)
       df_test = preprocessing.normalize(df_test)
       prediction = model.predict(df_test)       
       if prediction[0]==1:
          return render_template('HOME.html', prediction_text='This Headline/Tweet may increase market volatility' )
       else:
          return render_template('HOME.html', prediction_text='This Headline/Tweet may decrease market volatility' )
if __name__ == "__main__":
   app.run(port=9000,debug=False)


# In[ ]:




