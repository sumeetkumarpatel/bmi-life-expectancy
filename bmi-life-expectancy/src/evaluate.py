# Add import statements
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Assign the dataframe to this variable.
# Load the data (source: gapminder)
bmi_life_data = pd.read_csv('../data/bmi_and_life_expectancy.csv')
x_values = bmi_life_data[['BMI']]
y_values = bmi_life_data[['Life expectancy']]

# Make and fit the linear regression model
# Fit the model and Assign it to bmi_life_model
model = LinearRegression()
bmi_life_model = model.fit(x_values, y_values)

# Mak a prediction using the model
# Predict life expectancy for a BMI value of 21.07931
bmi_value = 21.07931
laos_life_exp = bmi_life_model.predict(bmi_value)
print(laos_life_exp)

# visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, bmi_life_model.predict(x_values))
plt.show()

