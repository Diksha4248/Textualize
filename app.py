#python .\main.py
from flask import Flask , render_template
from flask import request

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk


from tracemalloc import stop
import streamlit as st
import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from streamlit_option_menu import option_menu
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
sw=nltk.corpus.stopwords.words("english")

stopword = set(stopwords.words("english"))

nltk.download('vader_lexicon')

#function to clean and transform the user input which is in raw format
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    ps=PorterStemmer()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]','',text)
    text = re.sub('https?://\S+|www\.\S+','',text)
    text = re.sub('<.*?>+','',text)
    text = re.sub('[%s]' % re.escape(string.punctuation),'',text)
    text = re.sub('\n','',text)
    text = re.sub('\w*\d\w*','',text)
    text = [word for word in text.split(' ') if word not in stopword] 
    text = " ".join(text)
    
    return text    

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/d')
def d():
    #renders the diabetes prediction page template
    return render_template('contact.html')  

@app.route('/s')
def s():
    #renders the diabetes prediction page template
    return render_template('sentiment.html') 

@app.route('/sp')
def sp():
    #renders the diabetes prediction page template
    return render_template('spam_ham.html')            

@app.route('/home1')
def home1():
    return render_template("index.html") 


@app.route('/spam',methods=['POST'])
def spam_ham_predict():
    
    def transform1(txt1):
        txt2=tfidf1.fit_transform(txt1)
        return txt2.toarray()

    tfidf1=TfidfVectorizer(stop_words=sw,max_features=20)
    df1=pd.read_csv("Spam Detection.csv")
    df1.columns=["Label","Text"]
    x=transform1(df1["Text"])
    y=df1["Label"]
    x_train1,x_test1,y_train1,y_test1=train_test_split(x,y,test_size=0.1,random_state=0)
    model1=LogisticRegression()
    model1.fit(x_train1,y_train1)

    test_data = [x for x in request.form.values()][0]
    transformed_sent1=transform_text(test_data)
    vector_sent1=tfidf1.transform([transformed_sent1])
    prediction1=model1.predict(vector_sent1)[0]
    
    if prediction1 == "spam":
        data = "Spam"
    else:
        data = "Ham"    
        
    return render_template('spam_ham.html', prediction_text2='{}'.format(data))
       

#portion for hate speech prediction
@app.route('/di',methods=['POST'])
def hate_predict():
   
    dff = pd.read_csv("toxicity_en.csv")
    x = np.array(dff['text'])
    y = np.array(dff['is_toxic'])
    
    #bag of words
    cv = CountVectorizer() 
    x = cv.fit_transform(x)
    X_train , X_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state = 42)
    model = LogisticRegression().fit(X_train,y_train)
    test_data = [x for x in request.form.values()][0]
    he = cv.transform([test_data])
    y_prediction = model.predict(he)
    pred2 = y_prediction[0]
    print(pred2)

    if pred2 == "Toxic":
        data = "Toxic"
    else:
        data = "Non toxic"    
        
    return render_template('contact.html', prediction_text2='{}'.format(data))




@app.route('/se',methods=['POST'])
def sentiment_predict():
#Sentiment Analysis Prediction 

    def transform2(txt1):
        txt2=tfidf2.fit_transform(txt1)
        return txt2.toarray()

    tfidf2=TfidfVectorizer(stop_words=sw,max_features=20)
    df2=pd.read_csv("Sentiment Analysis.csv")
    df2.columns=["Text","Label"]
    x=transform2(df2["Text"])
    y=df2["Label"]
    x_train2,x_test2,y_train2,y_test2=train_test_split(x,y,test_size=0.1,random_state=0)
    # model2=LogisticRegression()
    # model2.fit(x_train2,y_train2)
    mnb = MultinomialNB()
    mnb.fit(x_train2,y_train2)
    y_pred2 = mnb.predict(x_test2)
    test_data = [x for x in request.form.values()][0]
    transformed_data=transform_text(test_data)
    vector_sent2=tfidf2.transform([transformed_data])
    prediction2=mnb.predict(vector_sent2)[0]

    if prediction2 == 0:
        data = "Negative Text"
    else:
        data = "Positive Text"    
        
    return render_template('sentiment.html', prediction_text2='{}'.format(data))


if __name__ == "__main__":
    app.run(debug=True)  