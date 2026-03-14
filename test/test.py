from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import os

from mlforge_studio.supervised_learning import MultipleLinearRegressionGD

file_path = os.path.join(os.path.dirname(__file__), "ecommerce.csv")
data = pd.read_csv(file_path)
X_train = data[['Time on App', 'Length of Membership']]
y_train = data['Yearly Amount Spent']
mymodel = MultipleLinearRegressionGD()
mymodel.fit(X_train, y_train)
skmodel = LinearRegression()
skmodel.fit(X_train, y_train)

sk_predictions = skmodel.predict(X_train)
predictions = mymodel.predict(X_train)


print("r2 score of my model: ", r2_score(y_pred=predictions, y_true=y_train))
print("r2 score of sklearn model: ", r2_score(y_pred=sk_predictions, y_true=y_train))
