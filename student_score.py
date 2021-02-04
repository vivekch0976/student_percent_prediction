# Import all necessory modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Load tha data and show it
data = pd.read_csv("student_scores.csv")
data.head()

# Make graphical representations
data.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Scored')
plt.show()

# Divide the data into attibutes and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values

# Split the data into test and traing data 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Import linear regression and fit the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Print the prediction of given data
y_pred = regressor.predict(X_test)
print(y_pred)

# Make prediction on own data
hours = [[9.25]]
own_pred = regressor.predict(hours)
print(own_pred)

# Analyse the algorithms
from sklearn import metrics
print ('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Erorr: ', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
