from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

d = load_breast_cancer()
X, y = d.data, d.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

m = SVC(kernel='linear')
m.fit(X_tr, y_tr)
y_pr = m.predict(X_te)

print(classification_report(y_te, y_pr, target_names=d.target_names))
