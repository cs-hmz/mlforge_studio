
import pandas as pd
import os

from mlforge_studio.supervised_learning import MultipleLinearRegressionGD

file_path = os.path.join(os.path.dirname(__file__), "ecommerce.csv")
data = pd.read_csv(file_path)
X_train = data[['Time on App', 'Length of Membership']]
y_train = data['Yearly Amount Spent']
mymodel = MultipleLinearRegressionGD()
mymodel.fit(X_train, y_train)

predictions = mymodel.predict(X_train)
print(predictions)