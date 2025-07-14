import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
from datetime import datetime
import os
import warnings
warnings.filterwarnings("ignore")

# -------------------------- PARAMETERS -----------------------------------

folder_path = 'data/all'  # <-- replace with your actual data path
results_file = 'median-dist-results.csv'

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

# Process a single file
def process_file(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Time'] = df['Date'].dt.strftime('%H:%M')

    split_index = int(0.7 * len(df))
    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()

    # Compute mean per (Weekday, Time) on training
    stats_table = train.groupby(['Weekday', 'Time'])['Volume'].agg(
        Mean='median'
    ).reset_index()

    # Merge into test
    test = test.merge(stats_table, on=['Weekday', 'Time'], how='left')
    test.dropna(subset=['Mean'], inplace=True)

    # Compute distance to mean as score
    test['score'] = (test['Volume'] - test['Mean']).abs()

    # Create combined labels
    test = test.assign(
        label_AND = (test['label_set_1'] & test['label_set_2']),
        label_OR = (test['label_set_1'] | test['label_set_2'])
    )

    label_sets = ['label_set_1', 'label_set_2', 'label_AND', 'label_OR']
    result_row = {'file': os.path.basename(file_path)}

    for idx, label in enumerate(label_sets):
        y_true = test[label]
        y_scores = test['score']

        # Find best threshold on scores to maximise F1
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_index = np.argmax(f1_scores)
        best_f1 = f1_scores[best_index]
        best_precision = precisions[best_index]
        best_recall = recalls[best_index]
        best_threshold = thresholds[best_index] if best_index < len(thresholds) else thresholds[-1]

        # ROC AUC on scores
        auc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else np.nan

        if idx == 0:
            result_row['best_c_label_set_1'] = best_threshold

        result_row[f'precision_{label}'] = best_precision
        result_row[f'recall_{label}'] = best_recall
        result_row[f'f1_{label}'] = best_f1
        result_row[f'auc_{label}'] = auc

    pd.DataFrame([result_row]).to_csv(results_file, mode='a', header=False, index=False)

# Process all files
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        process_file(file_path)

print("All files processed. Results saved to distance-results.csv.")
