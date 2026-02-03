# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the car price dataset and select features (engine size, horsepower, mileage) and target (price).
2. Split the data into training and testing sets, then standardize the feature values.
3. Train a Linear Regression model using the training data and predict prices for the test data.
4. Evaluate the model using performance metrics (MSE, RMSE, MAE, R²) and visualize results to check model accuracy.

## Program:
```
/*
 Program to implement linear regression model for predicting car prices and test assumptions.
Developed by: POPURI SAHITHYA
RegisterNumber:  212225240106
*/

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
df=pd.read_csv('CarPrice_Assignment.csv')
df.head()
X=df[['enginesize','horsepower','citympg','highwaympg']]
y=df['price']
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
model = LinearRegression()
model.fit(X_train_scaled,y_train)
y_pred=model.predict(X_test_scaled)
print('Name: Popuri Sahithya')
print('Reg No:212225240106')
print("MODEL COEFFICIENTS:")
for feature,coef in zip(X.columns,model.coef_):
    print(f"{feature:}:{coef:}")
print(f"{'Intercept':}:{model.intercept_:}")
print("\nMODEL PERFORMANCE:")
print(f"{'MSE':}:{mean_squared_error(y_test,y_pred):}")
print(f"{'RMSE':}:{np.sqrt(mean_squared_error(y_test,y_pred)):}")
print(f"{'MAE':}:{mean_absolute_error(y_test,y_pred):}")
print(f"{'R-squared':}:{r2_score(y_test,y_pred):}")
plt.figure(figsize=(10,5))
plt.scatter(y_test,y_pred,alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
plt.title("Linearity Check: Actual vs Predicted Prices")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.show()
residuals=y_test-y_pred
dw_test=sm.stats.durbin_watson(residuals)
print(f"\nDurbin-Watson Statistics: {dw_test:.2f}",
     "\n(Values close to 2 indicate no autocorrelation)")
plt.figure(figsize=(10,5))
sns.residplot(x=y_pred,y=residuals,lowess=True,line_kws={'color':'red'})
plt.title("Homoscedasticity Check: Residuals vs Predicted")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals ($)")
plt.grid(True)
plt.show()
fig, (ax1,ax2)= plt.subplots(1,2,figsize=(12,5))
sns.histplot(residuals,kde=True,ax=ax1)
ax1.set_title("Residuals Distribution")
sm.qqplot(residuals,line='45',fit=True,ax=ax2)
ax2.set_title("Q-Q Plot")
plt.tight_layout()
```

## Output:
<img width="304" height="312" alt="image" src="https://github.com/user-attachments/assets/eeb29615-f5c1-4397-84c0-63eb2042d27d" />

<img width="1113" height="591" alt="image" src="https://github.com/user-attachments/assets/3330e409-1e5a-4362-ae1c-c56dddd3bc20" />

<img width="1125" height="582" alt="image" src="https://github.com/user-attachments/assets/b5cae1eb-e456-4bf7-a5d1-1ed7084067ee" />

<img width="1229" height="508" alt="image" src="https://github.com/user-attachments/assets/bde3e3a2-3bf3-4fc3-88b6-0cfad436a0d3" />

## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
