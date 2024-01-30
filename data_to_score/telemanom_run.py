import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from tods.sk_interface.detection_algorithm.Telemanom_skinterface import TelemanomSKI
from extra_tools import normalize_data, df_percentage_of_ones
import time
import itertools


#model hyper_parameters
#smoothing_perc = [0.05]
smoothing_perc = [0.05]
epochs = [20]
window_size_ = [3, 5, 7, 10]
#window_size_ = [3]
#error_buffer = [3]
#error_buffer = [3]
dropout = [0.1]
#layers = [[10, 10, 10]]
layers = [[10, 10], [10, 10, 10], [10, 10, 10, 10]]
#l_s = [3, 5, 7, 10]
#l_s = [3]
n_predictions = [1]
#n_predictions = [1]

hyp_list = [smoothing_perc, epochs, window_size_, dropout, layers, n_predictions]

combinations = list(itertools.product(*hyp_list))


dir_list = os.listdir('data')
#dir_list = ['0185-E.csv', '2421-E.csv']


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


        transformer = TelemanomSKI(smoothing_perc = combinations[hyp][0], 
                                    epochs = combinations[hyp][1], 
                                    window_size_ = combinations[hyp][2], 
                                    error_buffer = combinations[hyp][2],
                                    dropout = combinations[hyp][3], 
                                    layers = combinations[hyp][4],
                                    l_s = combinations[hyp][2],
                                    n_predictions = combinations[hyp][5],
                                    contamination = contamination,
                                    )

        start1 = time.time()
        transformer.fit(X_train)
        end1 = time.time()

        prediction_score = transformer.predict_score(X_test)
        prediction_score = prediction_score.squeeze().tolist()

        prediction_labels_train = transformer.predict(X_train)
        prediction_labels_train = prediction_labels_train.squeeze().tolist()

        start2 = time.time()
        prediction_labels_test = transformer.predict(X_test)
        end2 = time.time()
        
        prediction_labels_test = prediction_labels_test.squeeze().tolist()
        
        train_time = round((end1 - start1),2)
        test_time = round((end2 - start2), 2)
        experiment = dir_list[location]+ '_' + str(combinations[hyp])
        
        answer = [prediction_labels_train,prediction_score,
            prediction_labels_test, train_time, test_time]
        results.setdefault(str(combinations[hyp]), answer)
        
    with open('telemanom_results/' + dir_list[location][0:-4] + '.txt', mode='w') as file:
        for key, value in results.items():
            file.write(f"{key}: {value}\n")

