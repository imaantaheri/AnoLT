import ast
import pandas as pd
from sklearn.metrics import precision_recall_curve as metrics
from sklearn.metrics import roc_auc_score
import os
import numpy as np
from extra_tools import roll_window

dir_list = os.listdir('data')

model = 'vae'

metric_data = []

for j in range(len(dir_list)):
    print(dir_list[j])
    ad = dir_list[j][0:-4]

    output_ad = model + '/' + model + '_results/' + ad + '.txt'
    data_ad = 'data/' + ad + '.csv'

    dict = {}
    with open(output_ad) as f:
        for line in f:
            key, value = line.strip().split(': ')
            dict[key] = value

    data = pd.read_csv(data_ad)

    label1 = data['label_set_1'].tolist()
    label2 = data['label_set_2'].tolist()

    label3 = [int(bit1) * int(bit2) for bit1, bit2 in zip(label1, label2)]
    label4 = [1 if bit1 == 1 or bit2 == 1 else 0 for bit1, bit2 in zip(label1, label2)]


    label_test_1 = label1[int(0.7 * len(data)):]
    label_test_2 = label2[int(0.7 * len(data)):]

    label_test_3 = label3[int(0.7 * len(data)):]
    label_test_4 = label4[int(0.7 * len(data)):]


    hyp = dict.keys()

    for i in hyp:
        dict[i] = dict[i].replace('nan', '0')
        test_pred_score = ast.literal_eval(dict[i])[1]

        if model in ['iforest', 'knn', 'ocsvm', 'lof', 'hbos', 'abod', 'cof', 'so_gaal', 'mo_gaal']:
            label_1 = roll_window(label_test_1, ast.literal_eval(i)[-1])
            label_2 = roll_window(label_test_2, ast.literal_eval(i)[-1])

            label_3 = roll_window(label_test_3, ast.literal_eval(i)[-1])
            label_4 = roll_window(label_test_4, ast.literal_eval(i)[-1])
        
        elif model in ['loda', 'discord']:
            label_1 = roll_window(label_test_1, ast.literal_eval(i)[0])
            label_2 = roll_window(label_test_2, ast.literal_eval(i)[0])

            label_3 = roll_window(label_test_3, ast.literal_eval(i)[0])
            label_4 = roll_window(label_test_4, ast.literal_eval(i)[0])
        
        elif model in ['pca', 'vae', 'ae']:
            label_1 = roll_window(label_test_1, ast.literal_eval(i)[0][0])
            label_2 = roll_window(label_test_2, ast.literal_eval(i)[0][0])

            label_3 = roll_window(label_test_3, ast.literal_eval(i)[0][0])
            label_4 = roll_window(label_test_4, ast.literal_eval(i)[0][0])

        elif model in ['deeplog', 'lstm', 'periodic_knn']:
            label_1 = label_test_1
            label_2 = label_test_2

            label_3 = label_test_3
            label_4 = label_test_4
        
        elif model in ['mp']:
            label_1 = roll_window(label_test_1, ast.literal_eval(i)[0])
            label_2 = roll_window(label_test_2, ast.literal_eval(i)[0])

            label_3 = roll_window(label_test_3, ast.literal_eval(i)[0])
            label_4 = roll_window(label_test_4, ast.literal_eval(i)[0])

            test_pred_score = test_pred_score[0:-(ast.literal_eval(i)[0]-1)]
        
        elif model in ['ar']:
            label_1 = label_test_1[ast.literal_eval(i)[0]:]
            label_2 = label_test_2[ast.literal_eval(i)[0]:]

            label_3 = label_test_3[ast.literal_eval(i)[0]:]
            label_4 = label_test_4[ast.literal_eval(i)[0]:]

        elif model in ['telemanom']:
            label_1 = label_test_1[ast.literal_eval(i)[2]+1:]
            label_2 = label_test_2[ast.literal_eval(i)[2]+1:]

            label_3 = label_test_3[ast.literal_eval(i)[2]+1:]
            label_4 = label_test_4[ast.literal_eval(i)[2]+1:]

        precisions1, recalls1, thresholds1 = metrics(label_1, 
                                                test_pred_score)
        
        precisions2, recalls2, thresholds2 = metrics(label_2, 
                                                    test_pred_score)
        
        precisions3, recalls3, thresholds3 = metrics(label_3, 
                                                    test_pred_score)
        
        precisions4, recalls4, thresholds4 = metrics(label_4, 
                                                    test_pred_score)
        
        np.seterr(divide='ignore', invalid='ignore')
        
        f1_scores1 = 2 * (precisions1 * recalls1) / (precisions1 + recalls1)
        f1_scores2 = 2 * (precisions2 * recalls2) / (precisions2 + recalls2)
        f1_scores3 = 2 * (precisions3 * recalls3) / (precisions3 + recalls3)
        f1_scores4 = 2 * (precisions4 * recalls4) / (precisions4 + recalls4)

        f1_scores1 = np.nan_to_num(f1_scores1)
        f1_scores2 = np.nan_to_num(f1_scores2)
        f1_scores3 = np.nan_to_num(f1_scores3)
        f1_scores4 = np.nan_to_num(f1_scores4)

        f1_score1 = max(f1_scores1)
        f1_score2 = max(f1_scores2)
        f1_score3 = max(f1_scores3)
        f1_score4 = max(f1_scores4)

        threshold1 = thresholds1[f1_scores1.argmax()]
        precision1 = precisions1[f1_scores1.argmax()]
        recall1 = recalls1[f1_scores1.argmax()]

        threshold2 = thresholds2[f1_scores2.argmax()]
        precision2 = precisions2[f1_scores2.argmax()]
        recall2 = recalls2[f1_scores2.argmax()]

        threshold3 = thresholds3[f1_scores3.argmax()]
        precision3 = precisions3[f1_scores3.argmax()]
        recall3 = recalls3[f1_scores3.argmax()]

        threshold4 = thresholds4[f1_scores4.argmax()]
        precision4 = precisions4[f1_scores4.argmax()]
        recall4 = recalls4[f1_scores4.argmax()]
        
        auc_roc1 = roc_auc_score(label_1, test_pred_score)
        auc_roc2 = roc_auc_score(label_2, test_pred_score)
        auc_roc3 = roc_auc_score(label_3, test_pred_score)
        auc_roc4 = roc_auc_score(label_4, test_pred_score)
        
        res_dict = {'loc': ad , 'hyp': i, 'auc_1': auc_roc1, 'pre_1': precision1, 'rec_1': recall1, 
                    'f_1': f1_score1, 'thr_1': threshold1,'auc_2': auc_roc2, 'pre_2': precision2, 'rec_2': recall2, 
                    'f_2': f1_score2, 'thr_2': threshold2, 'auc_3': auc_roc3, 'pre_3': precision3, 'rec_3': recall3, 
                    'f_3': f1_score3, 'thr_3': threshold3,'auc_4': auc_roc4, 'pre_4': precision4, 'rec_4': recall4, 
                    'f_4': f1_score4, 'thr_4': threshold4}
        
        metric_data.append(res_dict)

df1 = pd.DataFrame(metric_data) 
df1.to_csv(model + '/' + model + '_detailed.csv', index= False) 

metric_data_summary = []

for k in hyp:

    df_loc = df1[df1['hyp'].isin([k])].copy()

    PER1 = df_loc['pre_1'].mean()
    REC1 = df_loc['rec_1'].mean()
    F11 = df_loc['f_1'].mean()
    AUC1 = df_loc['auc_1'].mean()

    PER2 = df_loc['pre_2'].mean()
    REC2 = df_loc['rec_2'].mean()
    F12 = df_loc['f_2'].mean()
    AUC2 = df_loc['auc_2'].mean()

    F11_STD = df_loc['f_1'].std()
    F12_STD = df_loc['f_2'].std()

    PER3 = df_loc['pre_3'].mean()
    REC3 = df_loc['rec_3'].mean()
    F13 = df_loc['f_3'].mean()
    AUC3 = df_loc['auc_3'].mean()

    PER4 = df_loc['pre_4'].mean()
    REC4 = df_loc['rec_4'].mean()
    F14 = df_loc['f_4'].mean()
    AUC4 = df_loc['auc_4'].mean()

    F13_STD = df_loc['f_3'].std()
    F14_STD = df_loc['f_4'].std()

    summary = {'hyp': k, 'auc_1': AUC1, 'pre_1': PER1, 'rec_1': REC1, 'f_1': F11, 'f_1_std': F11_STD,
               'auc_2': AUC2, 'pre_2': PER2, 'rec_2': REC2, 'f_2': F12, 'f_2_std': F12_STD,
               'auc_3': AUC3, 'pre_3': PER3, 'rec_3': REC3, 'f_3': F13, 'f_3_std': F13_STD,
               'auc_4': AUC4, 'pre_4': PER4, 'rec_4': REC4, 'f_4': F14, 'f_4_std': F14_STD}

    metric_data_summary.append(summary)

df2 = pd.DataFrame(metric_data_summary)
df2.to_csv(model + '/' + model + '_summary.csv', index= False)

max_f1 = df2['f_1'].nlargest(3) 
first_max_f1 = max_f1.iloc[0]
second_max_f1 = max_f1.iloc[1]  
third_max_f1 = max_f1.iloc[2] 
max_rows = df2[df2['f_1'].isin([first_max_f1 ,second_max_f1, third_max_f1])] 

print(max_rows)
