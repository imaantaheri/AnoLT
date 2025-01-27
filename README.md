# AnoLT
Time Series Anomaly Detection (TAD) Benchmarking with labeled traffic count data

This repository represents data and code for the paper "On the Applicability of Time Series Anomaly Detection Methods to Real-World Traffic Volume Data".

## Requirements

TODS library need to be installed ([Check here](https://github.com/datamllab/tods)). 
For the TODS library, Linux system with Python 3.7 in a virtual envirement may be necessary. 

## Instructions

AnoLT dataset with ground truth labels are located in `data` folder. Data of each site is stored in a seperate `.csv` file. Different columns are provided to represent date, time, volume count, and labels assigned by the inspectors. 

To infer the exact spatial locations of these sites within Victoria, Australia visit and explore [here](https://vicroadsopendata-vicroadsmaps.opendata.arcgis.com/datasets/traffic-lights/).

To get performance metrics for each method, first, its related python code from `data_to_score` need to be executed. 
The code stores anomaly scores of the test set (last 30% samples in each location) into a folder named `[method name]_results`.

Based on the given set of hyperparameters at the begining of the code, every possible combination of them will be taken into account, and anomaly scores related to each combination will be stored in a `.txt` file per each location (Sample executed outputs for the AR method is uploaded for demonstration). 

For instance, the following part in the AR code can be used to adjust the hyperparameters: 

```
#model hyper_parameters
window_size = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
step_size = [1]
```

Then, derived anomaly scores of each method need to be copied to the `score_to_metrics` folder (as it is for the AR method).
`accuracy_extraction.py` need to be run, with `model = [model name]` at the begining, to get detailed and summary statistics of different metrics.

Final results for each method are also uploaded. The list of tested hyperparameters in the paper can be found in the `data_to_score` code and the summary results. 

Detailed results include accuracy metrics at every possible threshold level applied on anomaly scores.


There is also a file named `extra_tools.py` in some folders. There is no need to run it as it is only used in other scripts. 
