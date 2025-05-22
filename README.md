# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: THAMIZH KUMARAN S
RegisterNumber: 212223240166
*/
```
```python

import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])



```

## Output:

### Data Head:

![Screenshot 2025-05-21 220444](https://github.com/user-attachments/assets/3d095731-d1da-43c7-976e-17aa1677cb1e)


### Data Info:

![Screenshot 2025-05-21 220503](https://github.com/user-attachments/assets/a59f9c85-a97f-4db6-80f1-1789780845e3)


### isnull() sum():

![Screenshot 2025-05-21 220512](https://github.com/user-attachments/assets/e41092ad-8ca2-4f58-81eb-ba3b2dcd8b52)


### Data Head for salary:

![Screenshot 2025-05-21 220518](https://github.com/user-attachments/assets/1f921ec9-c0d8-4759-83c5-7ec00af779f0)


### Mean Squared Error :

![Screenshot 2025-05-21 220527](https://github.com/user-attachments/assets/8c8c1795-7fa7-42b6-bcda-1b8d9868ec07)


### r2 Value:

![Screenshot 2025-05-21 220532](https://github.com/user-attachments/assets/1397407a-d575-4893-a0e4-fb2cf4e071ac)


### Data prediction :

![Screenshot 2025-05-21 220543](https://github.com/user-attachments/assets/14135647-0638-43a6-9502-02de9571071e)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
