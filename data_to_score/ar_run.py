import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from tods.detection_algorithm.core.AutoRegOD import AutoRegOD
from extra_tools import normalize_data, df_percentage_of_ones
import time
import itertools



#model hyper_parameters 
window_size = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
#window_size = [3, 4]
step_size = [1]



hyp_list = [window_size, step_size]
combinations = list(itertools.product(*hyp_list))


#dir_list = os.listdir('data')
dir_list = ['0185-E.csv', '2421-E.csv']


for location in range(len(dir_list)):
    print(location)
    results = {}
    address = "data/" + dir_list[location]
    data = pd.read_csv(address)
    
    label_set_1 = 'label_set_1'

    contamination = df_percentage_of_ones(data, label_set_1)
    #contamination = 0.3
    split = int(round((0.7*len(data)),1))

    X_train , X_test = normalize_data(data, 'Volume')

    for hyp in range(len(combinations)):


        transformer = AutoRegOD(window_size = combinations[hyp][0],
                                step_size = combinations[hyp][1], 
                                contamination = contamination,)

        start1 = time.time()
        transformer.fit(X_train)
        end1 = time.time()

        prediction_score, X_left_inds, X_right_inds = transformer.decision_function(X_test)
        prediction_score = prediction_score.squeeze().tolist()

        prediction_labels_train, X_left_inds, X_right_inds = transformer.predict(X_train)
        prediction_labels_train = prediction_labels_train.squeeze().tolist()

        start2 = time.time()
        prediction_labels_test, X_left_inds, X_right_inds = transformer.predict(X_test)
        end2 = time.time()
        
        prediction_labels_test = prediction_labels_test.squeeze().tolist()
        
        train_time = round((end1 - start1),2)
        test_time = round((end2 - start2), 2)
        experiment = dir_list[location]+ '_' + str(combinations[hyp])
        
        answer = [prediction_labels_train,prediction_score,
            prediction_labels_test, train_time, test_time]
        results.setdefault(str(combinations[hyp]), answer)
        
    with open('ar_results/' + dir_list[location][0:-4] + '.txt', mode='w') as file:
        for key, value in results.items():
            file.write(f"{key}: {value}\n")

