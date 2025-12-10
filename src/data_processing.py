import csv
import numpy as np

def load_data(filepath):
    rows = []

    with open(filepath, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        
        for row in reader:
            rows.append(row)

    data = np.array(rows[1:], dtype=object)    
    columns = np.array(rows[0], dtype=object)  
    print("Loaded:", data.shape)
    return data, columns

def write_data(filepath, data, columns):
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(columns)
        writer.writerows(data)


def is_missing(x):
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    if isinstance(x, str) and (x.strip() == "" or x.strip().lower() == "null"):
        return True
    return False


def count_missing(data, columns, column_groups):
    missing_info = []
    for group_name, col_indices in column_groups.items():
        for idx in col_indices:
            col_name = columns[idx]
            col_data = data[:, idx]

            col_str = col_data.astype(str)
            
            is_missing_mask = (col_str == 'nan') | \
                              (col_str == 'null') | \
                              (col_str == '') | \
                              (col_str == 'None')
            
            missing_count = np.sum(is_missing_mask)

            missing_info.append((idx, group_name, col_name, missing_count))
    return missing_info


def one_hot(column_data):
    uniques = np.unique(column_data)
    encoding = np.zeros((len(column_data), len(uniques)), dtype=int)
    
    for i, val in enumerate(uniques):
        encoding[:, i] = (column_data == val).astype(int)
        
    return encoding, uniques
