#to Hamza, Hasfa, Paul
#import the following modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# get the dataset
dataset = pd.read_csv('Salary_Data.csv')

# select your metrix of features and dependant variables (X and y)
X = dataset.drop(['YearsExperience'], axis=1) # or you can simply use dataset.iloc[:,1:].values
y = dataset.iloc[:, 0].values

# creating our training and test data 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)


#now we create our regression model and train our model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


#now we test our model by trying to predict the test data
y_pred = regressor.predict(X_test)

#now we plot
plt.scatter(X_train, y_train, color = 'green')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')

plt.title('Cyber6:Lesson 001(training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()