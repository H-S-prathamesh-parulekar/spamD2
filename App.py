from flask import Flask, render_template, request
# from flask_debugtoolbar import DebugToolbarExtension
import pandas as pd
import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import smtplib
import time
import imaplib
import email
import traceback
app = Flask(__name__)
# app.debug = True
# app.config['SECRET_KEY'] = 'DontTellAnyone'
# toolbar = DebugToolbarExtension(app)
ORG_EMAIL = "@gmail.com"
FROM_EMAIL = "mailchecker07" + ORG_EMAIL
FROM_PWD = "vjti@1234"
SMTP_SERVER = "imap.gmail.com"
SMTP_PORT = 993

def read_email_from_gmail():
    try:
        mail = imaplib.IMAP4_SSL(SMTP_SERVER)
        mail.login(FROM_EMAIL, FROM_PWD)
        mail.select('inbox')

        data = mail.search(None, 'ALL')
        mail_ids = data[1]
        id_list = mail_ids[0].split()
        # first_email_id = int(id_list[0])
        latest_email_id = int(id_list[-1])
        first_email_id = latest_email_id - 1

        # print(first_email_id, latest_email_id)

        for i in range(latest_email_id, 2, -1):
            data = mail.fetch(str(i), '(RFC822)')
            # print(i)
            for response_part in data:
                arr = response_part[0]
                if isinstance(arr, tuple):

                    msg = email.message_from_string(str(arr[1], 'utf-8'))
                    maintype = msg.get_content_maintype()
                    email_subject = msg['subject']
                    email_message = msg['content']
                    email_from = msg['from']
                  #  print(msg)
                    if maintype == 'multipart':
                        for part in msg.get_payload():
                            if part.get_content_maintype() == 'text':
                                name = part.get_payload()
                                splitting = name.split('<div')
                                return splitting[0]
                              #  print(part.get_payload())
                                # print(f'After Trimming Whitespaces String =\'{part.get_payload().strip()}\'')
                                # print(1)
                            elif maintype == 'text':
                                # print(msg.get_payload())
                                name = msg.get_payload()
                                splitting = name.split('<div')
                                return splitting[0]

    except Exception as e:
        traceback.print_exc()
        print(str(e))


@app.route('/')
def home():
    data1 = read_email_from_gmail()
    df = pd.read_csv("spam2.csv", encoding="latin-1")
    #df.head()
    if 'Unnamed: 4' in df.columns:
        df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],
                axis=1, inplace=True)
    # Features and Labels
    df['label'] = df['class'].map({'ham': 0, 'spam': 1})
    df['label'] = df['label'].replace(0, np.nan)
    df['label'] = df['label'].fillna(0).astype(np.int64)
    df['label'] = df['label'].astype(np.int64, errors='ignore')
    X = df['message']
    y = df['label']
    # Extract Feature With CountVectorizer
    # cv = CountVectorizer()
    cv = TfidfVectorizer(min_df=1, stop_words='english')
    # X = cv.fit_transform(X)  # Fit the Data
    X = cv.fit_transform(X.values.astype('U'))  # Fit the Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    # Naive Bayes Classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
  
    if data1 is not None:
        data = [data1]
        vect = cv.transform(data).toarray()
        my_prediction2 = clf.predict(vect)

    #     # message = request.form['message']
    if data1 is not None:
        return render_template('index.html', prediction2=my_prediction2, data1=data1)
    else:
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
