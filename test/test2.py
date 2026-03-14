import pandas as pd
from mlforge_studio.supervised_learning import LinearRegressionClosedForm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os

file_path = os.path.join(os.path.dirname(__file__), "ecommerce.csv")
data = pd.read_csv(file_path)
X_train = data[['Time on App', 'Length of Membership']]
y_train = data['Yearly Amount Spent']

my_model = LinearRegressionClosedForm()
my_model.fit(X_train, y_train)

sklearn_model = LinearRegression()
sklearn_model.fit(X_train, y_train)

my_model_predictions = my_model.predict(X_train)
sklearn_model_predicitions = sklearn_model.predict(X_train)

print("r2 score of my model: ", r2_score(y_pred=my_model_predictions, y_true=y_train))
print("r2 score of sklearn model: ", r2_score(y_pred=sklearn_model_predicitions, y_true=y_train))
