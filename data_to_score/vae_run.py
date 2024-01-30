import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from tods.sk_interface.detection_algorithm.VariationalAutoEncoder_skinterface import VariationalAutoEncoderSKI
from extra_tools import normalize_data, df_percentage_of_ones, roll_window, roll_binary_window
import time
import itertools


#model hyper_parameters+

neurons = [
            [3, [2, 1],[2]], [3, [2],[2]],
            [4, [3, 2],[3]], [4, [3, 3],[3]],
            [5, [4, 3, 3],[3, 4]], [5, [4, 3, 2],[3, 4]],
            [7, [5, 4, 3],[4, 5]], [7, [5, 4, 2],[4, 5]], [7, [4, 3],[4]],
            [10, [7, 4],[4, 7]], [10, [5, 4, 3],[4, 5]], [10, [5, 4, 3, 2],[3, 4, 5]]
            ]

window_size = [3, 4, 5, 7, 10]


hyp_list = [neurons, window_size]
combinations = list(itertools.product(*hyp_list))


dir_list = os.listdir('data')
#dir_list = ['2421-E.csv']


for location in range(len(dir_list)):
    print(location)
    results = {}
    address = "data/" + dir_list[location]
    data = pd.read_csv(address)
    label_set_1 = 'label_set_1'

    X_train_main , X_test_main = normalize_data(data, 'Volume')
    
    for hyp in range(len(combinations)):
        print(combinations[hyp])
        X_train = roll_window(X_train_main, combinations[hyp][0][0])
        X_test = roll_window(X_test_main, combinations[hyp][0][0])
        
        label = roll_binary_window(data, label_set_1, combinations[hyp][0][0])
        contamination = df_percentage_of_ones(label, label_set_1)
        
        transformer = VariationalAutoEncoderSKI(encoder_neurons = combinations[hyp][0][1],
                                                decoder_neurons = combinations[hyp][0][2],
                                                contamination = contamination,
                                                preprocessing = False,
                                                verbose = 0
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
        
    with open('vae_results/' + dir_list[location][0:-4] + '.txt', mode='w') as file:
        for key, value in results.items():
            file.write(f"{key}: {value}\n")

