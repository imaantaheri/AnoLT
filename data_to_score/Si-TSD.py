import os
import numpy as np
import pandas as pd
from extra_tools import normalize_data, df_percentage_of_ones
import time
import itertools
import warnings
warnings.filterwarnings('ignore')
import heapq


#model hyper_parameters
n_neighbors = [3, 5, 7, 8, 9, 10]
#method = ['ave', 'max', 'med', 'k']

hyp_list = [n_neighbors]

combinations = list(itertools.product(*hyp_list))


dir_list = os.listdir('data')
#dir_list = ['2421-E.csv']



def cal_distance (target, values, k):
    #print(values)
    #print(target)
    distances = []
    for val in values:
        distance = abs(target - val)
        distances.append(distance)
    #print(distances)
    nearest = heapq.nsmallest(k, distances)
    return nearest


for location in range(len(dir_list)):
    print(location)
    results = {}
    address = "data/" + dir_list[location]
    data = pd.read_csv(address)
    label_set_1 = 'label_set_1'
    contamination = df_percentage_of_ones(data, label_set_1)

    df_X_train = data.iloc[0:int(0.7 * len(data))]
    df_X_train[['Date', 'Time']] = df_X_train['Date'].str.split(expand=True)
    
    df_X_test = data.iloc[int(0.7 * len(data)):]
    df_X_test = df_X_test.reset_index(drop = True)
    
    X_train_main , X_test_main = normalize_data(data, 'Volume')
    
    for hyp in range(len(combinations)):
        print(combinations[hyp])

        prediction_score = []
        prediction_labels_train = []
        prediction_labels_test = []

        start2 = time.time()
        for ind in range(len(X_test_main)):
            day_of_week = df_X_test['Weekday'].iloc[ind]
            time_of_day = df_X_test['Date'].iloc[ind]
            time_of_day = time_of_day[-8:]
            train_loc = df_X_train[(df_X_train['Weekday'].isin([day_of_week])) & 
                                   (df_X_train['Time'].isin([time_of_day]))].copy()
            train_index = train_loc.index.tolist()
            local = X_train_main[train_index].flatten().tolist()
            score = cal_distance(X_test_main[ind][0], local, combinations[hyp][0])
            score = score[-1]
            prediction_score.append(score)
        
        end2 = time.time()
        
        
        start1 = time.time()
        end1 = time.time()



        
        train_time = round((end1 - start1),2)
        test_time = round((end2 - start2), 2)
        experiment = dir_list[location]+ '_' + str(combinations[hyp])
        
        answer = [prediction_labels_train,prediction_score,
            prediction_labels_test, train_time, test_time]
        results.setdefault(str(combinations[hyp]), answer)
        
    with open('periodic_knn_results/' + dir_list[location][0:-4] + '.txt', mode='w') as file:
        for key, value in results.items():
            file.write(f"{key}: {value}\n")

