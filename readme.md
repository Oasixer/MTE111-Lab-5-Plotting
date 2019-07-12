### README / INSTRUCTIONS
0) Install dependencies obviously
1) Format your data to match data.csv - so columns 1-3 = strain vals for samples 1-3, and column 4 must be time (s)
2) Calculate your stress values for the three trials and paste them into the list below (stress_vals)
3) Set the values of initial_end to be the (starting at 1) index of the last data point that is in the initial stage for samples 1-3 respectively
4) Set the values of linear_end_relative to be the (NUMBER OF POINTS - 1) that are in the tertiary stage for samples 1-3 respectively
5) Set figure_3_use_error_bars to False probably, we used error bars in the positive direction bcz our data for fig 3 was lower than expected due to premature fractures
6) Run


---Note, the same instructions are in the top of the python file