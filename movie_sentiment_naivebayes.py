import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

d = pd.read_csv("https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv")[['label', 'tweet']].dropna()
X_tr, X_te, y_tr, y_te = train_test_split(d['tweet'], d['label'], test_size=0.2, random_state=42)

v = TfidfVectorizer()
X_tr_v = v.fit_transform(X_tr)
X_te_v = v.transform(X_te)

m = MultinomialNB()
m.fit(X_tr_v, y_tr)
y_pr = m.predict(X_te_v)

print(classification_report(y_te, y_pr))
