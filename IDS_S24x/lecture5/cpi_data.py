import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from bokeh.plotting import figure, show
from statsmodels.graphics.tsaplots import plot_acf, acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg


# Read data and convert date
ps_cpi = pd.read_csv("./CPI.csv")
ps_cpi['date'] = pd.to_datetime(ps_cpi['date'])  # Convert date values from string type to date type
ps_cpi['YearMonth'] = ps_cpi['date'].dt.strftime('%Y-%m')

# Pick one value from each month
cpi_last = ps_cpi.drop_duplicates('YearMonth', keep='last')  # Monthly CPI using last value of month
cpi_first = ps_cpi.drop_duplicates('YearMonth', keep='first')  # Monthly CPI using first value of month

# Merge last and first values to compare
cpi = cpi_last.merge(cpi_first, on='YearMonth', how='left', suffixes=('_last', '_first'))
cpi['diff_first_last'] = cpi['CPI_last'] - cpi['CPI_first']

# Extract relevant columns and split into training and test data
cpi = cpi[['YearMonth', 'CPI_last']]
cpi = cpi.rename(columns={'CPI_last': 'CPI'})
cpi_train = cpi[cpi.YearMonth < '2013-09'].copy()
cpi_test = cpi[cpi.YearMonth >= '2013-09'].copy()

# Add a new column 't' in units of month (1 for the first month, 2 for the second, etc.)
cpi_train['t'] = np.arange(1, len(cpi_train) + 1)  # t starting at 1
cpi_test['t'] = np.arange(len(cpi_train) + 1, len(cpi_train) + len(cpi_test) + 1)  # continuing t for test set

# Visualize the time series
p = figure(width=800, height=400, title="Monthly CPI")
p.line(cpi_train['t'], cpi_train.CPI, line_color="navy", line_width=2.5)
p.xaxis.axis_label = 'Months (t)'
p.yaxis.axis_label = 'CPI'
show(p)

# Fit a linear model
model = LinearRegression().fit(cpi_train[['t']], cpi_train.CPI)
coefficients = [model.coef_[0], model.intercept_]
print("The linear trend is given by F(t) = " + str(coefficients[0]) + "*t + (" + str(coefficients[1]) + ")")

# Predicting CPI using the linear model and plotting
linear_cpi_train = model.predict(cpi_train[['t']])
p = figure(width=800, height=400, title="CPI Time Series with Linear Trend")
p.line(cpi_train['t'], cpi_train.CPI, line_color="navy", line_width=2, legend_label="Original Data")
p.line(cpi_train['t'], linear_cpi_train, line_color="orange", line_width=2.5, legend_label="Linear Trend")
p.xaxis.axis_label = 'Months (t)'
p.yaxis.axis_label = 'CPI'
p.legend.location = 'bottom_right'
show(p)

# Define residuals
remaining = cpi_train.CPI - linear_cpi_train
linear_cpi_test = model.predict(cpi_test[['t']])
remaining_test = cpi_test.CPI - linear_cpi_test

# Plot residuals for training data
p = figure(width=800, height=400, title="Residuals of CPI Time Series")
p.line(cpi_train['t'], remaining, line_color="navy", line_width=2, legend_label="Detrended Data")
show(p)
print("The maximum residual value is: " + str(np.max(np.abs(remaining))))  # Report max absolute residual


# Determine the Lag
plot_acf(remaining)
plt.show()
plot_pacf(remaining)
plt.show()


# Step 1: Fit the AR(p) model
p = 2  # replace this with the lag value you identified earlier using the PACF plot
model_ar = AutoReg(remaining, lags=p)
model_ar_fit = model_ar.fit()

# Step 2: Extract the coefficients
alpha_coefficients = model_ar_fit.params

# Step 3: Print the coefficients (alpha_1 and alpha_2)
alpha_1 = alpha_coefficients[1]  # the coefficient for lag 1
alpha_2 = alpha_coefficients[2]  # the coefficient for lag 2

print(f"Alpha_1: {alpha_1:.2f}")
print(f"Alpha_2: {alpha_2:.2f}")