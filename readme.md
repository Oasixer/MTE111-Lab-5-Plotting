### README / INSTRUCTIONS
0) Install dependencies obviously

1) Format your data to match data.csv - so columns 1-3 = strain vals for samples 1-3, and column 4 must be time (s)

2) Calculate your stress values for the three trials and paste them into the stress_vals variable

3) Set the values of initial_end to be the (starting at 1) index of the last data point that is in the initial stage for samples 1-3 respectively

4) Set the values of linear_end_relative to be the (NUMBER OF POINTS - 1) that are in the tertiary stage for samples 1-3 respectively
---Note for steps 3,4: See Sample_Outputs.png to see what the values we used did in the plots

5) Set figure_3_use_error_bars to False probably, we used error bars in the positive direction bcz our data for fig 3 was lower than expected due to premature fractures
---Note: See Sample_Outputs.png for what the error bars in fig.3 look like

6) Run



---Note, the same instructions are in the top of the python file