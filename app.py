# Import statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,ConfusionMatrixDisplay
import os
import flask
from flask import render_template, request, session
from flask import jsonify
import requests, json
from flask_mysqldb import MySQL
import MySQLdb.cursors
from flask_session import Session
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from collections import Counter
import pickle


print("----Reading The Data----")
#file_path = 'newsData.txt'
data = pd.read_csv("Data/training_data.csv", nrows=10000)

print("----Initialize Vectorizer----")
# function to remove html elements from the reviews
def removeHTML(raw_text):
    clean_HTML = BeautifulSoup(raw_text, 'lxml').get_text() 
    return clean_HTML

# function to remove special characters and numbers from the reviews4961
def removeSpecialChar(raw_text):
    clean_SpecialChar = re.sub("[^a-zA-Z]", " ", raw_text)  
    return clean_SpecialChar

# function to convert all reviews into lower case
def toLowerCase(raw_text):
    clean_LowerCase = raw_text.lower().split()
    return( " ".join(clean_LowerCase)) 

# function to remove stop words from the reviews
def removeStopWords(raw_text):
    stops = set(stopwords.words("english"))
    words = [w for w in raw_text if not w in stops]
    return( " ".join(words))

model = pickle.load(open('Model/stack_election.pkl', 'rb'))

X = data['tweet']
Y = data['sentiment']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=30)

# X_training clean set
X_train_cleaned = []


for val in X_train:
    val = removeHTML(val)
    val = removeSpecialChar(val)
    val = toLowerCase(val)
    X_train_cleaned.append(val) 
    
# X_testing clean set
X_test_cleaned = []

for val in X_test:
    val = removeHTML(val)
    val = removeSpecialChar(val)
    val = toLowerCase(val)
    X_test_cleaned.append(val) 
    

tvec = TfidfVectorizer(use_idf=True,
strip_accents='ascii')

X_train_tvec = tvec.fit_transform(X_train_cleaned)

lr = LogisticRegression()
lr.fit(X_train_tvec, Y_train)

print("----Initializing The Flask Application----")
app = flask.Flask(__name__, template_folder='Templates')

print("----Database Connection----")
#code for connection
app.config['MYSQL_HOST'] = 'localhost'#hostname
app.config['MYSQL_USER'] = 'root'#username
app.config['MYSQL_PASSWORD'] = ''#password
app.config['MYSQL_DB'] = 'tweetapp'#database name

mysql = MySQL(app)


app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

@app.route('/')

@app.route('/main', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('homeindex.html'))

@app.route('/about', methods=['GET', 'POST'])
def about():
    if flask.request.method == 'GET':
        return(flask.render_template('about.html'))
    
@app.route('/services', methods=['GET', 'POST'])
def services():
    if flask.request.method == 'GET':
        return(flask.render_template('services.html'))
    
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if flask.request.method == 'GET':
        return(flask.render_template('contact.html'))
    
@app.route('/liveanalysis', methods=['GET', 'POST'])
def liveanalysis():
    if flask.request.method == 'GET':
        username = session.get("username")
        return(flask.render_template('liveanalysis.html', username=username))
    
@app.route('/searchdata', methods=['GET', 'POST'])
def searchdata():
    if flask.request.method == 'GET':
        username = session.get("username")
        return(flask.render_template('searchdata.html', username=username))
    
@app.route('/historypage', methods=['GET', 'POST'])
def historypage():
    if flask.request.method == 'GET':
        username = session.get("username")
        return(flask.render_template('historypage.html', username=username))
    
@app.route('/tweetpage', methods=['GET', 'POST'])
def tweetpage():
    if flask.request.method == 'GET':
        username = session.get("username")
        return(flask.render_template('tweetpage.html', username=username))
    
@app.route('/tweetanalytics', methods=['GET', 'POST'])
def tweetanalytics():
    if flask.request.method == 'GET':
        username = session.get("username")
        return(flask.render_template('tweetanalytics.html', username=username))

#User Login   
@app.route('/login', methods=['GET', 'POST'])
def login():
    if flask.request.method == 'GET':
        return(flask.render_template('login.html'))
    if flask.request.method == 'POST':
        msg=''
        if request.method == 'POST':
            phone    = request.form['phone']
            password = request.form['password']

            con = mysql.connect
            con.autocommit(True)
            cursor = con.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute('SELECT * FROM user_details WHERE phone = % s and password = %s', (phone, password,))
            result = cursor.fetchone()
            
         
        if result:
            msg = "1"
            session["userid"]     = result["user_id"]
            session["username"]   = result["user_name"]
        else:
           msg = "0"
    return msg
       
@app.route('/registeruser', methods=['GET', 'POST'])
def registeruser():
    if flask.request.method == 'GET':
        return(flask.render_template('register.html'))
    if flask.request.method == 'POST':
        msg=''
        #applying empty validation
        if request.method == 'POST':
            
            uname       = request.form['uname']
            email       = request.form['email']
            phone       = request.form['phone']
            password    = request.form['password']
            
            con = mysql.connect
            con.autocommit(True)
            cursor = con.cursor(MySQLdb.cursors.DictCursor)
            result = cursor.execute('INSERT INTO user_details VALUES (NULL, % s, % s, % s, % s, NULL)', (uname, email, phone, password,))
            mysql.connect.commit()

            #displaying message
            msg = '1'
    return msg



def extract_hashtags(tweet_content):
    words = re.findall(r'\b\w+\b', tweet_content.lower())
    common_words = [word for word in words if len(word) > 3 and word not in ['and', 'the', 'is', 'in', 'of', 'to', 'for', 'on', 'with']]
    top_common_words = Counter(common_words).most_common(30)
    hashtags = [f"#{word}" for word, _ in top_common_words]
    return hashtags

@app.route('/gettweethash', methods=['POST'])
def gettweethash():
    search = ''
    url = f"https://tweetcracks.000webhostapp.com/?search={search}"
    response = requests.get(url)
    if response.status_code == 200:
        tweets = response.json()
    combined_tweet_content = ' '.join(tweet['tweet_content'] for tweet in tweets)
    hashtags = extract_hashtags(combined_tweet_content)
    return jsonify(hashtags)
 
def analyze_sentiment(text):
    sentimentdt = [str(text)]
    prediction = lr.predict(tvec.transform(sentimentdt))[0]
    return prediction

import matplotlib.pyplot as plt
import random

@app.route('/getreleventtweet', methods=['POST'])
def get_relevant_tweets():
    search = request.form['searchdata'] 
    types = request.form['types'] 
    url = f"https://tweetcracks.000webhostapp.com/?search={search}"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        for tweet in data:
            cont = tweet["tweet_content"]
            
            cont = removeHTML(cont)
            cont = removeSpecialChar(cont)
            cont = toLowerCase(cont)

            prediction = analyze_sentiment(cont)
            
            if prediction == "positive":
                pred = "Positive"
            elif prediction == "negative":
                pred = 'Negative'
            else:
                pred = "Neutral"
                
            tweet["prediction"] = pred
        
        df = pd.DataFrame(data)
        sentiment_counts = df['prediction'].value_counts()

        # Custom colors for pie chart
        colors = {'Positive': '#44ce42', 'Negative': '#fc5a5a', 'Neutral': '#ffc542'}
        
        # Create pie chart
        plt.figure(figsize=(8, 8))
        plt.pie(sentiment_counts, labels=sentiment_counts.index, colors=[colors[sentiment] for sentiment in sentiment_counts.index], autopct='%1.1f%%', startangle=140)
        plt.title('Distribution of Sentiments')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        # Save chart image in static folder
        random_number = random.randint(1000, 9999)
        filename = f'pie_chart{random_number}.png'
        fname = 'static/'+filename
        plt.savefig(fname)
        
        totres = len(data)
        usid = session.get("userid")
        
        con = mysql.connect
        con.autocommit(True)
        cursor = con.cursor(MySQLdb.cursors.DictCursor)
        result = cursor.execute('INSERT INTO history VALUES (NULL, % s, % s, % s, % s, NULL)', (usid, search, totres, types,))
        mysql.connect.commit()
        
        out = {"tweet":data, "filename":filename}
    return jsonify(out)

@app.route('/getanalytics', methods=['POST'])
def getanalytics():
    search = '' 
    limit = random.randint(100, 200)
    url = f"https://tweetcracks.000webhostapp.com/?search={search}&limit={limit}"
    filename = ''
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        for tweet in data:
            cont = tweet["tweet_content"]
            
            cont = removeHTML(cont)
            cont = removeSpecialChar(cont)
            cont = toLowerCase(cont)

            prediction = analyze_sentiment(cont)
            
            if prediction == "positive":
                pred = "Positive"
            elif prediction == "negative":
                pred = 'Negative'
            else:
                pred = "Neutral"
                
            tweet["prediction"] = pred
            
        df = pd.DataFrame(data)
        sentiment_counts = df['prediction'].value_counts()

        # Custom colors for pie chart
        colors = {'Positive': '#44ce42', 'Negative': '#fc5a5a', 'Neutral': '#ffc542'}
        
        # Create pie chart
        plt.figure(figsize=(8, 8))
        plt.pie(sentiment_counts, labels=sentiment_counts.index, colors=[colors[sentiment] for sentiment in sentiment_counts.index], autopct='%1.1f%%', startangle=140)
        plt.title('Distribution of Sentiments')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        # Save chart image in static folder
        random_number = random.randint(1000, 9999)
        filename = f'pie_chart{random_number}.png'
        fname = 'static/'+filename
        plt.savefig(fname)

                    
    out = {"filename":filename}
    return jsonify(out)


@app.route('/historydata', methods=['GET', 'POST'])
def historydata():
    if flask.request.method == 'GET':
        return(flask.render_template('historypage.html'))
    if flask.request.method == 'POST':
            usid = session.get("userid")
            con = mysql.connect
            con.autocommit(True)
            cursor = con.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute('SELECT * FROM history WHERE userid = % s', (usid,))
            result = cursor.fetchall()
    
    return jsonify(result)
       

if __name__ == '__main__':
    app.run(debug=True)