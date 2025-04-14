from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

d = load_iris()
X, y = d.data, d.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=1)

m = DecisionTreeClassifier()
m.fit(X_tr, y_tr)
y_pr = m.predict(X_te)

print("Acc:", accuracy_score(y_te, y_pr))

plt.figure(figsize=(12, 8))
plot_tree(m, filled=True, feature_names=d.feature_names, class_names=d.target_names)
plt.show()
