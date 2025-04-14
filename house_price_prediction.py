import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

d = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")
X = d.drop('medv', axis=1)
y = d['medv']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

m = LinearRegression()
m.fit(X_tr, y_tr)
y_pr = m.predict(X_te)

print(f"MSE: {mean_squared_error(y_te, y_pr):.2f}")
print(f"R²: {r2_score(y_te, y_pr):.2f}")

plt.scatter(y_te, y_pr, c="blue")
plt.plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.grid()
plt.show()
