from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

d = load_digits()
X, y = d.data, d.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

m = KNeighborsClassifier(n_neighbors=3)
m.fit(X_tr, y_tr)
y_pr = m.predict(X_te)

print("Acc:", accuracy_score(y_te, y_pr))

fig, ax = plt.subplots(1, 5, figsize=(10, 3))
for i, a in enumerate(ax):
    a.imshow(X_te[i].reshape(8, 8), cmap='gray')
    a.set_title(f"Pred: {y_pr[i]}")
    a.axis("off")
plt.tight_layout()
plt.show()
