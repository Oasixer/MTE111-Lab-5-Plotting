import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import scipy.optimize as opt
import statsmodels.api as sm

'''
Code to plot some creep material stuff for Lab 5.

README / INSTRUCTIONS
0) Install dependencies obviously

1) Format your data to match data.csv - so columns 1-3 = strain vals for samples 1-3, and column 4 must be time (s)

2) Calculate your stress values for the three trials and paste them into the stress_vals variable

3) Set the values of initial_end to be the (starting at 1) index of the last data point that is in the initial stage for samples 1-3 respectively

4) Set the values of linear_end_relative to be the (NUMBER OF POINTS - 1) that are in the tertiary stage for samples 1-3 respectively
---Note for steps 3,4: See Sample_Outputs.png to see what the values we used did in the plots

5) Set figure_3_use_error_bars to False probably, we used error bars in the positive direction bcz our data for fig 3 was lower than expected due to premature fractures
---Note: See Sample_Outputs.png for what the error bars in fig.3 look like

6) Run
'''

# NOTE: Theoretically, any lab group should be able to use this code by only changing these 5 lines
# But your mileage may vary
data = pd.read_csv('data.csv') # Read data csv (creates pandas dataframe obj)
figure_3_use_error_bars = False # We used error bars on plot 3 because our values were kinda garbage. Defaulting this to False for you though.
stress_vals = [7.269933993, 8.185543999, 9.26234609] # Stress values, pasted in
initial_end = (9,3,2) # By inspection, end of primary stage - see readme @ top for instructions
linear_end_relative = (-3,0,0) # By inspection, end of secondary/linear stage - see readme @ top for instructions


'''
TOUCH ANYTHING BELOW HERE @ YOUR OWN RISK
'''


cols = (0,1,2) # Literally just the columns that contain strain values (aka columns 1-3 in the csv)
fractures = [data.iloc[:, col].last_valid_index() for col in cols]
linear_end_abs = [fractures[col]+linear_end_relative[col] for col in cols]
index_col = 3 # The column used for time - columns 0-2 are strain values (see the .csv file)

colors = [[(237/255, 0/255, 186/255), '#00b3ff', '#bdff61'], [(143/255, 11/255, 115/255), '#0969bd', '#48a623'], [(106/255, 6/255, 117/255), '#143787', '#126313']]

all_x = data.iloc[:, index_col].values.reshape(-1, 1)
all_y = [data.iloc[:, col].values.reshape(-1, 1) for col in cols]
initial_x = [data.iloc[:initial_end[col], index_col].values.reshape(-1, 1) for col in cols]
initial_y = [data.iloc[:initial_end[col], col].values.reshape(-1, 1) for col in cols]
tertiary_x = [data.iloc[linear_end_abs[col]-1:, index_col].values.reshape(-1, 1)  for col in cols]
tertiary_y = [data.iloc[linear_end_abs[col]-1:, col].values.reshape(-1, 1) for col in cols]
# Note, linear_x and linear_y are for the secondary / steady state stage
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

linear_fit_line = [linear_regressor[col].predict(linear_x[col]) for col in cols]  # make predictions

initial_lines = [None, None, None]
linear_points = [None, None, None]
linear_fits = [None, None, None]
final_lines = [None, None, None]
error_bars = [None, None, None]

for col in cols:
    initial_lines[col], = plt.plot(initial_x[col], initial_y[col], color=colors[0][col], linewidth=1, label=f'Sample.{col+1} Primary stage')
    linear_points[col] = plt.scatter(linear_x[col], linear_y[col], color=colors[1][col], s=30, marker='x', label=f'Sample.{col+1} Secondary stage data points')
    linear_fits[col], = plt.plot(linear_x[col], linear_fit_line[col], color=colors[1][col], label=f'Sample.{col+1} Secondary stage linear regression')
    final_lines[col], = plt.plot(tertiary_x[col], tertiary_y[col], color=colors[2][col], label=f'Sample.{col+1} Final stage')

first_legend = plt.legend([tuple(initial_lines), tuple(linear_fits), tuple(linear_points), tuple(final_lines)], ['Primary stage data points', 'Secondary stage fit line', 'Secondary stage data points', 'Final stage data points'], numpoints=1,
               handler_map={tuple: HandlerTuple(ndivide=None)}, loc='lower right')

# Add the legend manually to the current Axes.
ax = plt.gca().add_artist(first_legend)

 #Create another legend for the second line.
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
plt.ylabel('Strain (mm/mm)') # Figure 1 finished




plt.figure() # Create Figure 2 (log log plot of steady state strain rates vs engineering stress vals)
# Take the three steady state strain rates
# x axis is stress
steady_strain_rates = np.array([(linear_fit_line[col][-1][0]-linear_fit_line[col][0][0])/(linear_x[col][-1][0]-linear_x[col][0][0]) for col in cols]).reshape(-1,1)

ln_steady_strain_rates = np.log10(steady_strain_rates)

stress_vals = np.array(stress_vals).reshape(-1, 1) # x axis
ln_stress = np.log10(stress_vals)

points = plt.errorbar(stress_vals, steady_strain_rates, yerr=1.5*np.array(ols_standard_error), ecolor='black', marker='o', fmt='.', capsize=3, label=f'Sample.{col+1} Hi')

second_regressor = LinearRegression()
second_layer_regression = second_regressor.fit(ln_stress, ln_steady_strain_rates)
second_layer_fit_line = second_layer_regression.predict(ln_stress)

b = second_layer_regression.intercept_
x = 1
y = second_regressor.predict(np.array([[1]]))

m=(y-b)/x # Get slope of regression line in log space

ols_regressor2 = sm.OLS(ln_stress, ln_steady_strain_rates)
ols_results2 = ols_regressor2.fit()
ols_standard_error2=ols_results2.HC0_se[0]

log_fit = np.power(10, second_layer_fit_line)

fit_line_2_plt, = plt.plot(stress_vals, log_fit, color=colors[0][0], label='test')

plt.legend([(points), (fit_line_2_plt)], ['Data points, one per sample', f'Regression fit line [m={round(m[0][0], 5)}, b={round(b[0],4)}, S.D={round(ols_standard_error2, 6)}]'], numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)}, loc='best')
plt.xlabel('Stress (MPa)')
plt.yscale('log')
plt.xscale('log')
plt.yticks([0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01])
plt.ylabel('Steady state strain rate (mm/mm/t)')
plt.gca().yaxis.set_minor_formatter(plt.LogFormatter())#minor_thresholds=(1, 0.1)))
loc2 = plticker.LogLocator() # this locator puts ticks at regular intervals
plt.gca().yaxis.set_major_locator(loc2) # End of second figure




plt.figure() # add figure 3, plotting strain @ failure (x axis) vs engineering stress (y axis)
failure_strains = np.array([all_y[col][fractures[col]][0] for col in cols]).reshape(-1,1)

if figure_3_use_error_bars: # See instructions in readme
    plt.errorbar(failure_strains, stress_vals, xerr=[[0.001, 0.001, 0.001], [0.003, 0.01, 0.01]], xlolims=False, ecolor='black', marker='o', fmt='.', capsize=3, label=f'Data points with observation-based error bars')
else:
    plt.scatter(failure_strains, stress_vals, label='Data points, one per sample')

failure_strains = np.array([failure_strains[col][0] for col in cols])
stress_vals = np.array([stress_vals[col][0] for col in cols])

# Curve fit for plot 3
def func(x, a, b, c):
    return a * np.exp(-b * x) + c

popt, pcov = opt.curve_fit(func, failure_strains, stress_vals)
x_new = np.linspace(failure_strains[0], failure_strains[-1], 50)

def format_e(n): # Scientific notation helper function
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]

plt.plot(x_new, func(x_new, popt[0], popt[1], popt[2]), label=f'Exponential curve fit:\ny={format_e(np.round(popt[0]))}*exp(-{np.round(popt[1], 2)}*x) + {np.round(popt[2], 5)}')

plt.ylabel('Engineering stress (MPa)')
plt.xlabel('Strain at failure (mm/mm)')
plt.legend()

plt.show()