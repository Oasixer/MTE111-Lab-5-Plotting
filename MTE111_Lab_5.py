import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import scipy.optimize as opt;

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
for col in cols:
    linear_regressor[col].fit(linear_x[col], linear_y[col])  # perform linear regression

standard_dev = [data.iloc[initial_end[col]-1:linear_end_abs[col], col].values.std() for col in cols]
print(f'standard deviations: {standard_dev}')


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

textstr = f'Secondary stage standard deviations\n(error estimates for linear regression fit lines)\nSample 1: {round(standard_dev[0], 9)}\nSample 2: {round(standard_dev[1], 9)}\nSample 3: {round(standard_dev[2], 9)}'


plt.text(0, 0.14, textstr, fontsize=9,verticalalignment='top', bbox=props)

plt.xlabel('Time (s)')
plt.ylabel('Strain (mm/mm)')
plt.show()

