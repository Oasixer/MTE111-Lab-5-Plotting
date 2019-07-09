import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import scipy.optimize as opt
import statsmodels.api as sm

data = pd.read_csv('data.csv') # Outputs dataframe

initial_end = (9,3,2) # By inspection, end of primary stage
cols = (0,1,2)
fractures = [data.iloc[:, col].last_valid_index() for col in cols]
linear_end_relative = (-3,0,0) # By inspection, end of secondary/linear 
linear_end_abs = [fractures[col]+linear_end_relative[col] for col in cols]
index_col = 3

colors = [[(237/255, 0/255, 186/255), '#00b3ff', '#bdff61'], [(143/255, 11/255, 115/255), '#0969bd', '#48a623'], [(106/255, 6/255, 117/255), '#143787', '#126313']]

all_x = data.iloc[:, index_col].values.reshape(-1, 1)
all_y = [data.iloc[:, col].values.reshape(-1, 1) for col in cols]
initial_x = [data.iloc[:initial_end[col], index_col].values.reshape(-1, 1) for col in cols]
initial_y = [data.iloc[:initial_end[col], col].values.reshape(-1, 1) for col in cols]
tertiary_x = [data.iloc[linear_end_abs[col]-1:, index_col].values.reshape(-1, 1)  for col in cols]
tertiary_y = [data.iloc[linear_end_abs[col]-1:, col].values.reshape(-1, 1) for col in cols]
linear_x = [data.iloc[initial_end[col]-1:linear_end_abs[col], index_col].values.reshape(-1, 1)  for col in cols]
linear_y = [data.iloc[initial_end[col]-1:linear_end_abs[col], col].values.reshape(-1, 1)  for col in cols]

linear_regressor = [LinearRegression() for col in cols]  # create object for the class
ols_regressor = [None, None, None]
ols_results = [None, None, None]
ols_standard_error = [None, None, None]
relative_error = [None, None, None]
for col in cols:
    linear_regressor[col].fit(linear_x[col], linear_y[col])  # perform linear regression
    ols_regressor[col] = sm.OLS(linear_y[col], linear_x[col])
    ols_results[col] = ols_regressor[col].fit()
    ols_standard_error[col]=ols_results[col].HC0_se[0]
    relative_error[col] = ols_standard_error[col]/np.mean(linear_y[col])
    #print(ols_standard_error[col])

linear_fit_line = [linear_regressor[col].predict(linear_x[col]) for col in cols]  # make predictions

initial_lines = [None, None, None]
linear_points = [None, None, None]
linear_fits = [None, None, None]
final_lines = [None, None, None]
error_bars = [None, None, None]

print(linear_x[0][0][0])

for col in cols:
    initial_lines[col], = plt.plot(initial_x[col], initial_y[col], color=colors[0][col], linewidth=1, label=f'Sample.{col+1} Primary stage')
    linear_points[col] = plt.scatter(linear_x[col], linear_y[col], color=colors[1][col], s=30, marker='x', label=f'Sample.{col+1} Secondary stage data points')
    linear_fits[col], = plt.plot(linear_x[col], linear_fit_line[col], color=colors[1][col], label=f'Sample.{col+1} Secondary stage linear regression')
    final_lines[col], = plt.plot(tertiary_x[col], tertiary_y[col], color=colors[2][col], label=f'Sample.{col+1} Final stage')
    #error_bars[col] = plt.errorbar(np.delete(linear_x[col],np.s_[1:-1], axis=0), np.delete(linear_fit_line[col],np.s_[1:-1], axis=0), yerr=standard_dev[col], fmt='none')

first_legend = plt.legend([tuple(initial_lines), tuple(linear_fits), tuple(linear_points), tuple(final_lines)], ['Primary stage data points', 'Secondary stage fit line', 'Secondary stage data points', 'Final stage data points'], numpoints=1,
               handler_map={tuple: HandlerTuple(ndivide=None)}, loc='lower right')

# Add the legend manually to the current Axes.
ax = plt.gca().add_artist(first_legend)

 #Create another legend for the second line.
#precision = 4
plt.legend([tuple(i) for i in zip(initial_lines, linear_points, linear_fits, final_lines)], ['Sample 1', 'Sample 2', 'Sample 3'], numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)}, loc='right')

props = dict(boxstyle='round', facecolor='white', alpha=0.2)

steady_strain_rates = [(linear_fit_line[col][-1][0]-linear_fit_line[col][0][0])/(linear_x[col][-1][0]-linear_x[col][0][0]) for col in cols]

textstr = f'''Secondary stage fit line slopes
and standard errors
Sample 1: Slope {round(steady_strain_rates[0], 7)} S.D. {round(ols_standard_error[0], 7)}
Sample 2: Slope {round(steady_strain_rates[1], 7)} S.D. {round(ols_standard_error[1], 7)}
Sample 3: Slope {round(steady_strain_rates[2], 7)} S.D. {round(ols_standard_error[2], 7)}'''


plt.text(0, 0.14, textstr, fontsize=9,verticalalignment='top', bbox=props)

plt.xlabel('Time (s)')
plt.ylabel('Strain (mm/mm)')


plt.figure() # Create second figure

# Take the three steady state strain rates
# x axis is ln stress
steady_strain_rates = np.array([(linear_fit_line[col][-1][0]-linear_fit_line[col][0][0])/(linear_x[col][-1][0]-linear_x[col][0][0]) for col in cols]).reshape(-1,1)

#steady_strain_rates_error_bounds = np.array([[steady_strain_rates[col][0] - ols_standard_error[col] for col in cols], [steady_strain_rates[col][0]+ols_standard_error[col] for col in cols]])
ln_steady_strain_rates = np.log10(steady_strain_rates)
#ln_steady_strain_rates_error_bounds = np.log(steady_strain_rates_error_bounds)
#error_vals = np.array([[ln_steady_strain_rates[col][0] - ln_steady_strain_rates_error_bounds[0][col] for col in cols],[ln_steady_strain_rates_error_bounds[1][col]-ln_steady_strain_rates[col][0] for col in cols]])

stress_vals = np.array([7.269933993, 8.185543999, 9.26234609]).reshape(-1, 1) # x axis
ln_stress = np.log10(stress_vals)

#ln_points = plt.errorbar(ln_stress, ln_steady_strain_rates, ecolor='black', yerr=error_vals, marker='o', fmt='.', capsize=3, label=f'Sample.{col+1} Hi')
points = plt.errorbar(stress_vals, steady_strain_rates, yerr=1.5*np.array(ols_standard_error), ecolor='black', marker='o', fmt='.', capsize=3, label=f'Sample.{col+1} Hi')

second_regressor = LinearRegression()
second_layer_regression = second_regressor.fit(ln_stress, ln_steady_strain_rates)
second_layer_fit_line = second_layer_regression.predict(ln_stress)

b = second_layer_regression.intercept_
x = 1
y = second_regressor.predict(np.array([[1]]))

#(y-b)/x = m
m=(y-b)/x # Get slope of regression line in log space

ols_regressor2 = sm.OLS(ln_stress, ln_steady_strain_rates)
ols_results2 = ols_regressor2.fit()
ols_standard_error2=ols_results2.HC0_se[0]

print(ols_results2.HC0_se)


log_fit = np.power(10, second_layer_fit_line)

fit_line_2_plt, = plt.plot(stress_vals, log_fit, color=colors[0][0], label='test')
print((log_fit-ols_standard_error[1]).reshape(1, -1))

#plt.fill_between(np.array([7.269933993, 8.185543999]), (log_fit-ols_standard_error[0]).reshape(1, -1)[0][0], (log_fit[1]))

plt.legend([(points), (fit_line_2_plt)], ['Data point for each sample', f'Regression fit line [m={round(m[0][0], 5)}, b={round(b[0],4)}, S.D={round(ols_standard_error2, 6)}]'], numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)}, loc='best')
plt.xlabel('Stress (MPa)')
plt.yscale('log')
plt.xscale('log')
plt.yticks([0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01])
plt.ylabel('Steady state strain rate (mm/mm/t)')
#plt.gca().xaxis.set_minor_formatter(plt.NullFormatter())
plt.gca().yaxis.set_minor_formatter(plt.LogFormatter())#minor_thresholds=(1, 0.1)))
#loc = plticker.MultipleLocator(base=0.1) # this locator puts ticks at regular intervals
loc2 = plticker.LogLocator() # this locator puts ticks at regular intervals
#plt.gca().xaxis.set_major_locator(loc)
plt.gca().yaxis.set_major_locator(loc2)

plt.figure() # add figure
# plotting strain @ failure (x axis) vs engineering stress (y axis)
#print(all_y)
failure_strains = np.array([all_y[col][fractures[col]][0] for col in cols]).reshape(-1,1)
print(failure_strains)
plt.scatter(failure_strains, stress_vals, label='Data points, one per sample')
third_regressor = LinearRegression()
third_layer_regression = third_regressor.fit(failure_strains, stress_vals)
third_layer_fit_line = third_layer_regression.predict(failure_strains)
plt.plot(failure_strains, third_layer_fit_line, label='Fit line')

plt.ylabel('Engineering stress (MPa)')
plt.xlabel('Strain at failure (mm/mm)')
plt.legend()

plt.show()

#7.269933993