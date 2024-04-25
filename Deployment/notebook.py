import pandas as pd
import numpy as np
data = pd.read_csv('data.csv')

# Split name to type & model col
def get_laptop_type(laptop_name):
    if 'Gaming' in laptop_name:
        return 'Gaming Laptop'
    else:
        return 'Laptop'

data['type'] = data['name'].apply(get_laptop_type)
data['model'] = data['name'].str.replace('Gaming Laptop', '').str.replace('Laptop', '')

# HANDEL RAM FEATURE
data['Ram'] = data['Ram'].str.replace('GB', '')
data['Ram'] = data['Ram'].astype(float)


# HANDEL ROM FEATURE
def convert_to_gb(value):
    if 'TB' in value:
        return (float(value.replace('TB', '')) * 1000)
    else:
        return float(value.replace('GB', ''))


data['ROM'] = data['ROM'].apply(convert_to_gb)


# split CPU to core & thread col
def split_cpu(cpu_val):
    if ',' in cpu_val:
        parts = cpu_val.split(',')
        if len(parts) == 2:
            part1 = parts[0].strip()
            part2 = parts[1].strip()
        else:
            part1 = parts[0].strip()
            part2 = None
    else:
        if 'cores' in cpu_val.lower() or 'core' in cpu_val.lower():
            part1 = cpu_val
            part2 = None

        else:
            part1 = None
            part2 = cpu_val
    return part1, part2


data['core'], data['thread'] = zip(*data['CPU'].apply(split_cpu))

## HANDEL thread FEATURE
data['thread'] = data['thread'].str.replace('Threads', '')
data['thread'] = data['thread'].astype(float)

# fill nulls in thread col
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=3, metric='nan_euclidean')
data_reshaped = data['thread'].values.reshape(-1, 1)
data_imputed = imputer.fit_transform(data_reshaped)
data_imputed = data_imputed.astype(int)
data['thread'] = data_imputed

# Fill nulls in the core col
m_price = data['price'].median()
closest_price_row = data.iloc[(data['price'] - m_price).abs().idxmin()]
data['core'] = data['core'].fillna(closest_price_row['core'])


# Extract GPU information into three features
import re
data['GPU_Brand'] = data['GPU'].str.extract(r'(amd|intel|nvidia)')
data['GPU_VRAM'] = data['GPU'].str.extract(r'(\d+gb|\d+mb)')
data['GPU_Type'] = data['GPU'].str.extract(r'(?:amd|intel|nvidia)\s*(?:\d+gb|\d+mb)?\s*(.*)', flags=re.IGNORECASE)[
    0].str.strip()

# Replace NaN values in the extracted features with "other"
data['GPU_Brand'].fillna("other", inplace=True)
data['GPU_VRAM'].fillna("other", inplace=True)
data['GPU_Type'].fillna("other", inplace=True)

#  spec_rating
# Ram
# Ram_type
# ROM
# display_size
# resolution_width
# resolution_height
#  model
# core
# thread
# GPU_Type


# DROP UNNESSARY COLUMNS
data.drop(
    columns=['Unnamed: 0.1', 'Unnamed: 0', 'brand', 'name', 'processor', 'CPU', 'GPU', 'OS', 'ROM_type', 'warranty',
             'GPU_Brand', 'GPU_VRAM', 'type'], inplace=True)

# ENCODE
import pickle

lst = ['Ram_type', 'core', 'GPU_Type', 'model']
for col in lst:
    value_counts = data.groupby(col)['price'].mean()
    value_counts = value_counts.sort_values(ascending=True)
    label_encoder = {cat: i for i, cat in enumerate(value_counts.index)}
    data[col] = data[col].replace(label_encoder).astype(float)
    # Save the encoder to a file with the column name
    with open(f'{col}_encoder.pkl', 'wb') as file:
        pickle.dump(label_encoder, file)


# SPLIT DATA
from sklearn.model_selection import train_test_split
target = data['price']
features = data.drop('price', axis=1)

from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split

# Split your data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize the QuantileTransformer
scaler_quantile = QuantileTransformer(output_distribution='normal', n_quantiles=min(1000, len(X_train)))

# Fit the transformer on the training data
scaler_quantile.fit(X_train)

# Transform the training and test data
X_train_scaled = scaler_quantile.transform(X_train)
X_test_scaled = scaler_quantile.transform(X_test)

# Check for NaN values
print("NaN in X_train_scaled:", np.isnan(X_train_scaled).any())
print("NaN in X_test_scaled:", np.isnan(X_test_scaled).any())


# MODLING
from sklearn.tree import DecisionTreeRegressor
dtc = DecisionTreeRegressor()
dtc.fit(X_train, y_train)
y_pred_dtc = dtc.predict(X_test)

import pickle

pickle.dump(dtc, open('Decision.pkl', 'wb'))  # this serializes the object
pickle.dump(scaler_quantile, open('scaler.pkl', 'wb'))