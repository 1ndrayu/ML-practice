import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

d = pd.read_csv("https://raw.githubusercontent.com/blastchar/telco-customer-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv")
d = d.dropna().drop(['customerID'], axis=1)
for c in d.select_dtypes(include=['object']).columns:
    d[c] = LabelEncoder().fit_transform(d[c])

X = d.drop('Churn', axis=1)
y = d['Churn']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

m = RandomForestClassifier(n_estimators=100)
m.fit(X_tr, y_tr)
y_pr = m.predict(X_te)

print(classification_report(y_te, y_pr))
