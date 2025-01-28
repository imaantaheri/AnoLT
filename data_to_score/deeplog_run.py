import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from tods.sk_interface.detection_algorithm.DeepLog_skinterface import DeepLogSKI
from extra_tools import normalize_data, df_percentage_of_ones
import time
import itertools
from multiprocessing import Pool, cpu_count


#model hyper_parameters
hidden_size = [32, 64]
#hidden_size = [32]
 
epochs= [20]
dropout_rate = [0.1]
window_size = [3,5,7,10]
#window_size = [7]
stacked_layers = [2, 3, 4]
#stacked_layers = [3]

hyp_list = [hidden_size, epochs, window_size, dropout_rate, stacked_layers]
combinations = list(itertools.product(*hyp_list))


#dir_list = os.listdir('data')
dir_list = ['4218-W.csv', '2101-S.csv', '2434-E.csv', '2417-E.csv', '2100-W.csv', '2421-E.csv']


def process_dataset(location):
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


        transformer = DeepLogSKI(hidden_size = combinations[hyp][0], 
                                 epochs = combinations[hyp][1], 
                                 window_size = combinations[hyp][2], 
                                 dropout_rate = combinations[hyp][3],
                                 stacked_layers = combinations[hyp][4], 
                                 contamination = contamination, 
                                 verbose = 0)

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
        
    with open('deeplog_results/' + dir_list[location][0:-4] + '.txt', mode='w') as file:
        for key, value in results.items():
            file.write(f"{key}: {value}\n")

if __name__ == '__main__':
    with Pool(processes=cpu_count()) as pool:
        pool.map(process_dataset, range(len(dir_list)))



        

