# Predictive-Maintenance

During their lifetime, aircraft components are susceptible to degradation, which affects directly their reliability and performance. 
This machine learning project will be directed to provide a framework for predicting the aircraft’s remaining useful life (RUL) based on the entire life cycle data 
in order to provide the necessary information to schedule maintenance. 

Dataset:
Experimental Scenario Data sets consists of multiple multivariate time series. Each data set is further divided into training and test subsets. Each time series is from a 
different engine i.e., the data can be considered to be from a fleet of engines of the same type. Each engine starts with different degrees of initial wear and manufacturing
variation which is unknown to the user. This wear and variation is considered normal, i.e., it is not considered a fault condition. There are three operational settings that 
have a substantial effect on engine performance. These settings are also included in the data. The data is contaminated with sensor noise.

1. FD001 (Train, Test, RUL)
2. FD002 (Train, Test, RUL)
3. FD003 (Train, Test, RUL)
4. FD004 (Train, Test, RUL)

The dataset is transformed from *.txt into set of features and target columns in *.csv files
Features:
Engine, Cycles, Sensor Data 1-24

Target:
Remaining Cycles
