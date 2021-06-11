from flask import Flask, render_template, request
from flask_debugtoolbar import DebugToolbarExtension
import pandas as pd
import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.debug = True
app.config['SECRET_KEY'] = 'DontTellAnyone'
toolbar = DebugToolbarExtension(app)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv("spam2.csv", encoding="latin-1")
    #df.head()
    if 'Unnamed: 4' in df.columns:
        df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    # Features and Labels
    df['label'] = df['class'].map({'ham': 0, 'spam': 1})
    df['label']  = df['label'].replace(0,np.nan)
    df['label'] = df['label'].fillna(0).astype(np.int64)
    df['label'] = df['label'].astype(np.int64,errors='ignore') 
    X = df['message']
    y = df['label']
    # Extract Feature With CountVectorizer
    # cv = CountVectorizer()
    cv = TfidfVectorizer(min_df=1,stop_words='english')
    # X = cv.fit_transform(X)  # Fit the Data
    X = cv.fit_transform(X.values.astype('U'))  # Fit the Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # Naive Bayes Classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('index.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run()