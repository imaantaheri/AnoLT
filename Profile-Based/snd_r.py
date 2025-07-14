import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from datetime import datetime
import os
import warnings
warnings.filterwarnings("ignore")

# -------------------------- PARAMETERS -----------------------------------

# Folder containing CSV files
folder_path = 'data/all'  # <-- replace with your actual data path

# Grid search parameters
c_start = 0.05
c_end = 5.0
c_step = 0.001

# Results file
results_file = 'SND-robust-results.csv'

# --------------------------------------------------------------------------

# Initialize results file
if not os.path.exists(results_file):
    pd.DataFrame(columns=[
        'file', 'best_c_label_set_1',
        'precision_label_set_1', 'recall_label_set_1', 'f1_label_set_1', 'auc_label_set_1',
        'precision_label_set_2', 'recall_label_set_2', 'f1_label_set_2', 'auc_label_set_2',
        'precision_label_AND', 'recall_label_AND', 'f1_label_AND', 'auc_label_AND',
        'precision_label_OR', 'recall_label_OR', 'f1_label_OR', 'auc_label_OR'
    ]).to_csv(results_file, index=False)

# Function to process each file
def process_file(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Time'] = df['Date'].dt.strftime('%H:%M')

    split_index = int(0.7 * len(df))
    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()

    iqr_table = train.groupby(['Weekday', 'Time'])['Volume'].agg(
        Q1=lambda x: x.quantile(0.25),
        Q3=lambda x: x.quantile(0.75),
        Median='median'
    ).reset_index()
    iqr_table['IQR'] = iqr_table['Q3'] - iqr_table['Q1']

    test = test.merge(iqr_table, on=['Weekday', 'Time'], how='left')
    test.dropna(subset=['Median', 'IQR'], inplace=True)

    test = test.assign(
        label_AND = (test['label_set_1'] & test['label_set_2']),
        label_OR = (test['label_set_1'] | test['label_set_2'])
    )

    label_sets = ['label_set_1', 'label_set_2', 'label_AND', 'label_OR']
    c_values = np.arange(c_start, c_end + c_step, c_step)

    result_row = {'file': os.path.basename(file_path)}

    for idx, label in enumerate(label_sets):
        best_f1 = -1
        best_result = None
        for c in c_values:
            upper = test['Median'] + c * test['IQR']
            lower = test['Median'] - c * test['IQR']
            test['pred'] = ~test['Volume'].between(lower, upper)
            
            y_true = test[label]
            y_pred = test['pred']
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if len(np.unique(y_true)) < 2:
                auc = np.nan
            else:
                auc = roc_auc_score(y_true, y_pred)
            
            if f1 > best_f1:
                best_f1 = f1
                best_result = {
                    'c': c,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc
                }
        
        if idx == 0:
            result_row['best_c_label_set_1'] = best_result['c']
        result_row[f'precision_{label}'] = best_result['precision']
        result_row[f'recall_{label}'] = best_result['recall']
        result_row[f'f1_{label}'] = best_result['f1']
        result_row[f'auc_{label}'] = best_result['auc']

    pd.DataFrame([result_row]).to_csv(results_file, mode='a', header=False, index=False)

# Process all files in folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        process_file(file_path)

print("All files processed. Results being saved live to results.csv.")
