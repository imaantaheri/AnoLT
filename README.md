# AnoLT
Time Series Anomaly Detection (TAD) Benchmarking with labeled traffic count data

This repository represents data and code for the paper "AnoLT: On the Applicability of Time Series Anomaly Detection Methods to Real-World Road Traffic Count Data".

## Requirements

TODS library need to be installed ([Check here](https://github.com/datamllab/tods)). 
For the TODS library, Linux system with Python 3.7 in a virtual envirement may be necessary. 

## Instruction

AnoLT dataset with ground truth labels are located in `data` folder. 

To get performance metrics for each method, first, its related python code from `data_to_score` need to be executed. 
The code stores anomaly scores of the test set (last 30% samples in each location) into a folder named `[method name]_results`.
Based on the given set of hyperparameters at the begining of the code, every possible combination of them will be taken into account, and anomaly scores related to each combination will be stored in a `.txt` file per each location. (Sample executed outputs for the AR method is uploaded for demonstration). 

Then, derived anomaly scores of each method need to be copied to the `score_to_metrics` folder (as it is for the AR method).  

