import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 1. Load data
file_path = 'data/all/2100-E.csv'  # <-- replace this with your actual data path
df = pd.read_csv(file_path)

# Parse date and extract time-of-day
df['Date'] = pd.to_datetime(df['Date'])
df['Time'] = df['Date'].dt.strftime('%H:%M')  # time-of-day as string

# 2. Split into train/test (70% time-based split)
split_index = int(0.7 * len(df))
train = df.iloc[:split_index].copy()
test = df.iloc[split_index:].copy()

# 3. Compute IQR for each Weekday+Time

iqr_table = train.groupby(['Weekday', 'Time'])['Volume'].agg(
    Q1=lambda x: x.quantile(0.25),
    Q3=lambda x: x.quantile(0.75)
).reset_index()
iqr_table['IQR'] = iqr_table['Q3'] - iqr_table['Q1']

# Merge IQR table back into test data
test = test.merge(iqr_table, on=['Weekday', 'Time'], how='left')

# 4. Generate two additional label sets
test['label_AND'] = (test['label_set_1'] & test['label_set_2'])
test['label_OR'] = (test['label_set_1'] | test['label_set_2'])

# 5. Function to compute metrics for given constant c
def evaluate_iqr(c, test, label_col):
    upper = test['Q3'] + c * test['IQR']
    lower = test['Q1'] - c * test['IQR']
    
    test['pred'] = ~test['Volume'].between(lower, upper)
    
    y_true = test[label_col]
    y_pred = test['pred']
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_pred)
    except:
        auc = np.nan

    return precision, recall, f1, auc

# 6. Grid search over multiple c values
from collections import defaultdict

results = defaultdict(list)
label_sets = ['label_set_1', 'label_set_2', 'label_AND', 'label_OR']
c_values = np.arange(0.1, 4, 0.001)

for label in label_sets:
    for c in c_values:
        precision, recall, f1, auc = evaluate_iqr(c, test, label)
        results['label_set'].append(label)
        results['c'].append(c)
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['f1'].append(f1)
        results['auc'].append(auc)

# 7. Collect results
results_df = pd.DataFrame(results)

# 8. Find best c for each label set
final_results = []
for label in label_sets:
    subset = results_df[results_df['label_set'] == label]
    best_row = subset.loc[subset['f1'].idxmax()]
    final_results.append(best_row)

final_results_df = pd.DataFrame(final_results).reset_index(drop=True)

# 9. Display final results
print(final_results_df[['label_set', 'c', 'precision', 'recall', 'f1', 'auc']])

# 10. Visualization for label_set_1
best_c_label1 = final_results_df.loc[final_results_df['label_set'] == 'label_set_1', 'c'].values[0]

# Generate predictions with best c for label_set_1
upper = test['Q3'] + best_c_label1 * test['IQR']
lower = test['Q1'] - best_c_label1 * test['IQR']
test['pred'] = ~test['Volume'].between(lower, upper)

# Prepare masks
true_pos = (test['pred'] == 1) & (test['label_set_1'] == 1)
false_pos = (test['pred'] == 1) & (test['label_set_1'] == 0)
false_neg = (test['pred'] == 0) & (test['label_set_1'] == 1)
true_neg = (test['pred'] == 0) & (test['label_set_1'] == 0)

# Plot
plt.figure(figsize=(15,6))
plt.plot(test['Date'], test['Volume'], color='lightgray', label='Volume')
plt.scatter(test['Date'][true_pos], test['Volume'][true_pos], color='green', label='True Positives')
plt.scatter(test['Date'][false_pos], test['Volume'][false_pos], color='red', label='False Positives')
plt.scatter(test['Date'][false_neg], test['Volume'][false_neg], color='orange', label='False Negatives')
plt.legend()
plt.title(f'IQR Anomaly Detection for label_set_1 (c={best_c_label1})')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.show()
