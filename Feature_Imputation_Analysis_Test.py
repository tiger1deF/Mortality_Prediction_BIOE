import pandas as pd
from pathlib import Path
import time
import tqdm
from datetime import datetime
import os
import sys
import polars as pl
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_curve, confusion_matrix, roc_curve
from sklearn.metrics import auc as skauc
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from fancyimpute import IterativeImputer as MICE
import numpy as np

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
def basic_model(format_str = "%Y-%m-%d %H:%M:%S", impute = True):

    # Obtain discharge/death/admission times for label processing
    data = pd.read_csv('dataset/ADMISSIONS.csv')
    chart_data = pd.read_csv('dataset/CHARTEVENTS.csv',)
    admit_times = data['ADMITTIME']
    discharge_times = data['DISCHTIME']
    death_times = data['DEATHTIME']
    subject_ids = data['SUBJECT_ID']
    
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

    # Conversion dict for item IDs - This is already efficient
    conversion_data = pd.read_csv('dataset/D_ITEMS.csv')
    id_conversion = {i: j for i, j in zip(conversion_data['ITEMID'], conversion_data['LABEL'])}
    
    # Prepare the feature names and the subject IDs
    feature_names = {'Sodium': 'Sodium (135-148)', 'Phosphorous': 'Phosphorous(2.7-4.5)', 
                     'Magnesium': 'Magnesium (1.6-2.6)', 'Glucose': 'Glucose (70-105)'}
    feature_ids = {v: k for k, v in id_conversion.items() if v in feature_names.values()}
    
    # Filter chart_data only for the relevant ITEMIDs
    chart_data = chart_data[chart_data['ITEMID'].isin(list(feature_ids.values()))]
    
    # Convert CHARTTIME to datetime once, outside of the loop
    chart_data['CHARTTIME'] = pd.to_datetime(chart_data['CHARTTIME'], format=format_str)
    
    # Initialize the dictionary for features
    features = {feature: [] for feature in feature_names.values()}
    
    for subject_id, admit_time in tqdm.tqdm(zip(subject_ids, admit_times), total=len(subject_ids), desc='Obtaining subject chart features...'):
        
        # Filter data for the current subject
        subject_data = chart_data[chart_data['SUBJECT_ID'] == subject_id]
    
        # Pivot the table: ITEMID becomes columns, values are 'VALUE', index is 'CHARTTIME'
        subject_data_pivoted = subject_data.pivot(index='CHARTTIME', columns='ITEMID', values='VALUE')
    
        # Rename columns using id_conversion
        subject_data_pivoted.rename(columns=id_conversion, inplace=True)
    
        # Check if all required features are present in this row
        try:
            required_features_present = subject_data_pivoted[feature_names.values()].notnull().all(axis=1)
            # Filter rows where all required features are present
            complete_cases = subject_data_pivoted[required_features_present]
        except:
            for feature in feature_names.values():
                features[feature].append(None)
        
        # If there are complete cases, sort by 'CHARTTIME' and take the earliest
        if not complete_cases.empty:
            earliest_complete_record = complete_cases.sort_index().iloc[0]
    
            # Store the earliest complete record's data
            for feature in feature_names.values():
                features[feature].append(earliest_complete_record[feature])
        else:
            # Handle cases with no complete data (you might want to append NaN or some indicator)
            for feature in feature_names.values():
                features[feature].append(None)    
    features = pd.DataFrame(features)
    
    # Data preprocessing and imputation (if enabled)
    print(f'Original data length: {len(data)}')
    
    def convert_to_float_or_none(val):
      try:
          return float(val)
      except:
          return None
        
    if impute == True:
        scaler = StandardScaler()
        imputer = MICE()
        numpy_features = np.array([[convert_to_float_or_none(item) for item in row] for row in features.to_numpy()])
        numpy_features = scaler.fit_transform(numpy_features)
        imputed_data = imputer.fit_transform(numpy_features)
        features = pd.DataFrame(imputed_data, columns=features.columns, index=features.index)
    
    # Creates analytical dataframe and equalizes labels
    data = pd.DataFrame({
        'Death':death_labels,
    })
    data = pd.concat((features, data), axis = 1)
    data = data.sample(frac = 1.0)
    
    if impute == False:
        data = data.dropna()
        
    data = equalize_data(data)  
    print(f'Processed data length: {len(data)}')
    

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
    
