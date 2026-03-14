import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlforge_studio.supervised_learning import SimpleLinearRegressionGD
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os

file_path = os.path.join(os.path.dirname(__file__), "salary_data.csv")
data = pd.read_csv(file_path)
X_ = data['YearsExperience']
X_train = data[['YearsExperience']]
y_train = data['Salary']

my_model = SimpleLinearRegressionGD(epochs=60000)
sk_model = LinearRegression()

my_model.fit(X_, y_train)
sk_model.fit(X_train, y_train)

my_predictions = []
for i in range(len(X_)):
    my_predictions.append(my_model.predict(X_[i]))
my_predictions = np.array(my_predictions)

sk_predictions = sk_model.predict(X_train)

plt.scatter(data['YearsExperience'], data['Salary'], color="black", label="Data")
plt.plot(list(range(1, 11)), [my_model.predict(x) for x in range(1, 11)], color="red", label="Custom Model")
plt.plot(list(range(1, 11)), [sk_model.predict([[x]])[0] for x in range(1, 11)], color="blue", linestyle="--", label="Sklearn Model")
plt.legend()
plt.show()