from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np
from nltk.stem.porter import PorterStemmer
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle


## Definitions
def remove_pattern(input_txt,pattern):
    r = re.findall(pattern,input_txt)
    for i in r:
        input_txt = re.sub(i,'',input_txt)
    return input_txt
def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")),3)*100




data = pd.read_csv("sentiment.tsv",sep = '\t')
data.columns = ["label","body_text"]
# Features and Labels
data['label'] = data['label'].map({'pos': 0, 'neg': 1})
data['tidy_tweet'] = np.vectorize(remove_pattern)(data['body_text'],"@[\w]*")
tokenized_tweet = data['tidy_tweet'].apply(lambda x: x.split())
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) 
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
data['tidy_tweet'] = tokenized_tweet
data['body_len'] = data['body_text'].apply(lambda x:len(x) - x.count(" "))
data['punct%'] = data['body_text'].apply(lambda x:count_punct(x))
X = data['tidy_tweet']
y = data['label']
# Extract Feature With CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data
X = pd.concat([data['body_len'],data['punct%'],pd.DataFrame(X.toarray())],axis = 1)
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
## Using Classifier
clf = LogisticRegression()
clf.fit(X.values,y)

pickle.dump(clf, open("model.pkl", "wb"))