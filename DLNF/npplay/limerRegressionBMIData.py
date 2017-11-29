import numpy as np
import pandas as pn


bmi_life_data =  pn.read_csv("./bmi.csv") 

from sklearn.linear_model import LinearRegression
bmi_life_model = LinearRegression()

x_values = bmi_life_data[['BMI']]
y_values = bmi_life_data[['Life expectancy']]

bmi_life_model.fit(x_values, y_values)


# Mak a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
laos_life_exp = bmi_life_model.predict([[21.07931]])
