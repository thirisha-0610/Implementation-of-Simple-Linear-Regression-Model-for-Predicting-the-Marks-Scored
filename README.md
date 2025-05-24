# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Gather data consisting of two variables. Input- a factor that affects the marks and Output - the marks scored by students

2.Plot the data points on a graph where x-axis represents the input variable and y-axis represents the marks scored

3.Define and initialize the parameters for regression model: slope controls the steepness and intercept represents where the line crsses the y-axis

4.Use the linear equation to predict marks based on the input Predicted Marks = m.(hours studied) + b

5.For each data point calculate the difference between the actual and predicted marks

5.Adjust the values of m and b to reduce the overall error. The gradient descent algorithm helps update these parameters based on the calculated error

6.Once the model parameters are optimized, use the final equation to predict marks for any new input data

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Thirisha A
RegisterNumber:  212223040228
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


df= pd.read_csv('/content/student_scores.csv')


df.head()
df.tail()


X=df.iloc[:,:-1].values
X
y=df.iloc[:,-1].values
y


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/2,random_state=0)


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)


y_pred=regressor.predict(X_test)
y_pred


y_test


import matplotlib.pyplot as plt
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")


plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")


mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:

Head Values:

![420779592-21c8d22f-fdb9-4ba5-934d-6bc0fe7d06c5](https://github.com/user-attachments/assets/a272d873-1b33-4b69-9335-4ad22bce042f)

Tail Values:

![420780044-4c2a9a86-eff9-4fbe-ba6f-ef7f41770d31](https://github.com/user-attachments/assets/2526572a-2f0d-4d81-a15d-e9011da00a2a)

X values:

![420780171-a20853d2-3777-4fe1-bf84-5f9fd78ac419](https://github.com/user-attachments/assets/dae9cad8-2bb7-418a-9007-6030a51086ed)

y Values:
Predicted Values

![420780306-2ce266e1-4996-4548-9318-b97297b245f8](https://github.com/user-attachments/assets/d4030723-f4e8-4b34-b21e-e94026ba0a9b)

Actual Values:

![420780570-1113439d-8305-4bfd-a686-d936ad6f462a](https://github.com/user-attachments/assets/a216d6e3-eb79-44ce-95db-2c7532f8024f)

Training Set:

![420780712-e4cfa84d-aedf-4d15-9ec5-3bc58e6e6eab](https://github.com/user-attachments/assets/256fe67c-d283-4877-9e7b-71e66643575e)

Testing Set:

![420780858-88a41f37-235b-43e0-a4b7-fa6f3901ad60](https://github.com/user-attachments/assets/2ffeaafa-834c-43e5-b49f-81901b6a55d5)

MSE, MAE and RMSE :

![420781138-fe7e70da-fe97-4abe-a52a-11e8dd4aa53c](https://github.com/user-attachments/assets/e5bab4ca-ffab-492c-9131-6dab80462121)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
