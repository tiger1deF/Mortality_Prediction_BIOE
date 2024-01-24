import pandas as pd
from pathlib import Path
import time
import tqdm
from datetime import datetime
import os
import sys
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_curve, confusion_matrix, roc_curve
from sklearn.metrics import auc as skauc
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Function for equalizing data
def equalize_data(dataframe, label = 'Death'):
    true_data = dataframe[dataframe[label] == 1]
    false_data = dataframe[dataframe[label] == 0]
    true_length, false_length = len(true_data), len(false_data)
    
    # Equalizes labels
    if true_length > false_length:
        sample_fraction = false_length / true_length
        true_data = true_data.sample(frac = sample_fraction)
    else:
        sample_fraction = true_length / false_length
        false_data = false_data.sample(frac = sample_fraction)
        
    # Recombines data
    data = pd.concat((true_data, false_data), axis = 0)
    return data
    

# Basic ML model testing
def basic_model(format_str = "%Y-%m-%d %H:%M:%S"):

    # Obtain discharge/death/admission times for label processing
    data = pd.read_csv('dataset/ADMISSIONS.csv')
    admit_times = data['ADMITTIME']
    discharge_times = data['DISCHTIME']
    death_times = data['DEATHTIME']
    
    # Obtains labels
    death_labels = []
    hospital_stay = []
    for admit, dis, death in tqdm.tqdm(zip(admit_times, discharge_times, death_times), total = len(admit_times), desc = 'Processing mortality'):
        if isinstance(death, float):
            hospital_stay.append((datetime.strptime(dis, format_str) - datetime.strptime(admit, format_str)).total_seconds())
            death_labels.append(0)
        else:
            hospital_stay.append((datetime.strptime(death, format_str) - datetime.strptime(admit, format_str)).total_seconds())
            death_labels.append(1)
            
    # Creates analytical dataframe and equalizes labels
    data = pd.DataFrame({
        'Hospital Stay':hospital_stay,
        'Death':death_labels,
    })
    data = equalize_data(data)
    data = data.sample(frac = 1.0)
    
    # Data prep
    train_data, test_data = train_test_split(data, test_size = 0.2, random_state=42)
    X_train, y_train = train_data.iloc[:, :-1].to_numpy(), train_data.iloc[:, -1].to_numpy()
    X_test, y_test = test_data.iloc[:, :-1].to_numpy(), test_data.iloc[:, -1].to_numpy()
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Performs random forest analysis
    clf = RandomForestClassifier(n_estimators = 1000, random_state=42)
    clf.fit(X_train, y_train)
    

    # Performs prediction and predictions analytics
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    print(f'RF Accuracy: {accuracy}')
        
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc = skauc(fpr, tpr)
    
    print(f'RF AUC: {auc}')
    plt.figure()
    plt.plot(fpr, tpr, label = f'Random Forest (AUC: {auc})', color = 'red')
    plt.plot([0, 1], [0, 1], label = 'Random Classifier', linestyle = '--', color = 'blue')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Random Forest Classifier on Predicting Mortality based on Hospital Stay')
    plt.savefig('RF_AUC.png')
  
  
# Main runtime
if __name__ == '__main__':
    basic_model()
    
    
    
'''
import dask.dataframe as dd
import gc
from pandarallel import pandarallel

def optimize_memory(df):
    # Convert columns with few unique values to 'category' data type
    for col in df.columns:
        if df[col].nunique() / len(df[col]) < 0.5:  # Arbitrary threshold
            df[col] = df[col].astype('category')
    return df

def check_for_duplicates(df, index_col):
    if df.index.duplicated().any():

        # Adjust groupby to explicitly specify the 'observed' parameter based on your data's needs
        df = df.groupby(df.index, observed=False).agg(lambda x: x.tolist() if len(set(x)) > 1 else x.iloc[0])
        
    return df

def process_data(dataset_location=Path('dataset'), chunksize=1000000):
    start = time.perf_counter()
    files = [i for i in dataset_location.glob('*.csv')]
    
    # Read the base dataframe
    base_file = dataset_location / 'ADMISSIONS.csv'
    if base_file.exists():
        base_df = pd.read_csv(base_file, low_memory=False)
        base_df = optimize_memory(base_df)
        base_index_col = 'SUBJECT_ID' if 'SUBJECT_ID' in base_df.columns else base_df.columns[0]
        base_df.set_index(base_index_col, inplace=True)
        base_df = check_for_duplicates(base_df, base_index_col)
        print(f"Base file read: {base_file}")
    else:
        print("Base file (ADMISSIONS.csv) not found!")
        return None

    for file in tqdm.tqdm(files, desc="Processing files"):
        if file != base_file:
            existing_columns = set(base_df.columns)
            print(f"Reading and merging file: {file}")
            if 'CHARTEVENTS' in file.stem:
                total_rows = sum(1 for row in open(file, 'r', encoding='utf-8', errors='ignore')) - 1  # subtract 1 for the header
                total_chunks = (total_rows // chunksize) + (1 if total_rows % chunksize else 0)
                
                for chunk in tqdm.tqdm(pd.read_csv(file, chunksize=chunksize), total=total_chunks, desc=f"Processing chunks of {file.stem}"):
                    chunk = optimize_memory(chunk)
                    merge_index_col = base_index_col if base_index_col in chunk.columns else list(set(base_df.columns) & set(chunk.columns))[0]
                    chunk.set_index(merge_index_col, inplace=True)
                    chunk = check_for_duplicates(chunk, merge_index_col)

                    # Identify and rename duplicate columns with a unique suffix in chunk
                    duplicate_columns = [col for col in chunk.columns if col in existing_columns]
                    unique_suffix = f"_{file.stem}"
                    rename_dict = {col: f"{col}{unique_suffix}" for col in duplicate_columns}
                    chunk.rename(columns=rename_dict, inplace=True)

                    # Join the chunk to the base_df
                    base_df = base_df.join(chunk, how='left', rsuffix=unique_suffix)
                    
            else:
                df = pd.read_csv(file, low_memory=False)
                df = optimize_memory(df)
                merge_index_col = base_index_col if base_index_col in df.columns else list(set(base_df.columns) & set(df.columns))[0]
                
                if merge_index_col:
                    df.set_index(merge_index_col, inplace=True)
                    df = check_for_duplicates(df, merge_index_col)

                    # Identify and rename duplicate columns with a unique suffix in df
                    duplicate_columns = [col for col in df.columns if col in existing_columns]
                    unique_suffix = f"_{file.stem}"
                    rename_dict = {col: f"{col}{unique_suffix}" for col in duplicate_columns}
                    df.rename(columns=rename_dict, inplace=True)
    
                    # Join the df to the base_df
                    base_df = base_df.join(df, how='left', rsuffix=unique_suffix)
                
                else:
                    print(f"No shared columns for merging found in {file.name}. Skipping this file.")

            print(f"After merging with {file.stem}: {len(base_df)} rows")
    
    base_df.reset_index(inplace=True)
    print(f"Final merged dataframe length: {len(base_df)}")
    end = time.perf_counter()
    print(f'Total runtime: {round(end - start, 4)} seconds')
    
    base_df.to_csv('Mortality_FULL.csv')
    return base_df
    
# Processes a more efficient merged dataset 
def process_individual_data(dataset_location=Path('dataset')):
    start = time.perf_counter()
    files = [i for i in dataset_location.glob('*')]
    
    def read_and_aggregate(file, group_by='SUBJECT_ID', agg_func=lambda x: list(x)):
        print(f"Reading file: {file}")
        df = pd.read_csv(file, low_memory=False)

        if group_by not in df.columns:
            print(f"'{group_by}' column not found in {file.name}. Skipping this file.")
            return None

        df = df.groupby(group_by).agg(agg_func).reset_index()
        return df

    dfs = {
        file.stem: read_and_aggregate(file)
        for file in files
        if any(keyword in file.stem.upper() for keyword in 
               ['ADMISSION', 'CALLOUT', 'NOTEE', 'MICRO', 'OUTPUT', 'DIAGNOSES_', 'CPTE', 'LABE', 'CHARTEVENTS', 'PRESCRIPTIONS'])
    }

    dfs = {k: v for k, v in dfs.items() if v is not None}

    if dfs:
        print(f"Dataframes to merge: {list(dfs.keys())}")  # Print the keys of the dataframes

        merged_df = None
        for k, df in dfs.items():
            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on='SUBJECT_ID', how='inner', suffixes=('', f'_{k}'))
            
            print(f"After merging with {k}: {len(merged_df)} rows")  # Debugging print

        print(f"Final merged dataframe length: {len(merged_df)}")
    else:
        print("No dataframes to merge.")
        return None

    end = time.perf_counter()
    print(f'Total runtime: {round(end - start, 4)} seconds')
    return merged_df
'''