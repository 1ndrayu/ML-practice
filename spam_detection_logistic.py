import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

d = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv", sep='\t', names=["label", "msg"])
d['label'] = d['label'].map({'ham': 0, 'spam': 1})

X_tr, X_te, y_tr, y_te = train_test_split(d['msg'], d['label'], test_size=0.2, random_state=42)

v = TfidfVectorizer()
X_tr_v = v.fit_transform(X_tr)
X_te_v = v.transform(X_te)

m = LogisticRegression()
m.fit(X_tr_v, y_tr)
y_pr = m.predict(X_te_v)

print(classification_report(y_te, y_pr))
