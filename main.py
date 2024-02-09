import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


data = pd.read_csv('data.csv')


X = data.drop(["Position","Salary"] , axis = 1) 
y = data["Salary"]

plt.scatter(X, y)
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()


poly = PolynomialFeatures(degree = 3) # 3 darajali polinom yaratiyapmiz
X_poly = poly.fit_transform(X) # X ning qiymani polinom uchun moslashtirilyapti


X_train , X_test , y_train , y_test = train_test_split(X_poly , y, test_size = 0.10 , random_state = 40) # o'qitish uchun x va y qiymatlarni beryapmiz va test qiliib ko'rish uchun tasodifiy qiymatlarni beryapmiz 

model = LinearRegression()
model = model.fit(X_train , y_train) # Chiziqli regressiyani o'qityapmiz

y_pred = model.predict(X_test) # Bashorat qilyapmiz y qiymatlarni

MSE = mean_squared_error(y_test , y_pred)
print("MSE = {}".format(MSE)) # O'rtacha kvadratik xatolik

model_linear = LinearRegression() # Chiziqli regressiya obyekt olyapmiz
model_linear = model_linear.fit(X, y) # Chiziqli regressiya o'qityapmiz


y_pred_linear = model_linear.predict(X) # y Bashorat qilinyapti

y_pred_all = model.predict(X_poly) # Poliminal regressiya bashorat qilinyapti

plt.scatter(X, y , label = "Distribution" , color = "navy")
plt.plot(X, y_pred_all, label = "Polynomial Regression" , color = "red" , linewidth = 5)
plt.plot(X, y_pred_linear, label = "Linear Regression" , color = "green")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.legend()
plt.show()