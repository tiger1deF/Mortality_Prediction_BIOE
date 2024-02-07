import pandas as pd
from pathlib import Path
import time
import tqdm
from datetime import datetime
import os
import sys
import polars as pl
import math
from sklearn.preprocessing import StandardScaler, normalize, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_recall_curve, confusion_matrix, roc_curve
from sklearn.metrics import auc as skauc
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from fancyimpute import IterativeImputer as MICE
from sklearn.compose import ColumnTransformer
import numpy as np
from multiprocessing import Pool
import multiprocessing
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# Function for analyzing features
def analyze_features(features_scaled, feature_names, y, k=5):
    """
    Analyze features using K-means, Linear Regression, and PCA.
    
    :param features_scaled: Scaled feature numpy array
    :param feature_names: List of feature names
    :param y: True labels
    :param k: Number of clusters for K-means
    """
    
    # K-means clustering for feature analysis
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features_scaled)
    centroids = kmeans.cluster_centers_
    centroid_feature_std = np.std(centroids, axis=0)
    feature_importance_kmeans = pd.Series(centroid_feature_std, index=feature_names).sort_values(ascending=False)
    print("Feature importance based on K-means clustering:")
    print(feature_importance_kmeans)
    print("\n" + "-"*50 + "\n")
    feature_importance_kmeans.to_csv('Feature_Importance_Kmeans.csv')
    
    # Linear Regression for feature importance
    lr = LinearRegression()
    print(y.count(0))
    print(y.count(1))
    lr.fit(features_scaled, y)
    feature_importance_lr = pd.Series(np.abs(lr.coef_), index=feature_names).sort_values(ascending=False)
    print("Feature importance based on Linear Regression:")
    print(feature_importance_lr)
    print("\n" + "-"*50 + "\n")
    feature_importance_lr.to_csv('Feature_Importance_Regression.csv')
    
    # Perform PCA
    pca = PCA()
    pca.fit(features_scaled)
    
    # Analyze explained variance ratio to decide on the number of components
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    num_components = np.argmax(cumulative_variance >= 0.95) + 1  # for 95% cumulative variance
    
    print(f"Proportion of components to retain for 95% cumulative variance: {num_components/len(feature_names)}")
    
    # Inspect the PCA loadings for the top components
    pca_loadings = pd.DataFrame(pca.components_[:num_components], columns=feature_names)
    
    # Absolute loadings (contribution) of features to each principal component
    abs_loadings = np.abs(pca_loadings)
    
    # Sum of absolute loadings across retained components to estimate feature importance
    feature_importance = abs_loadings.sum(axis=0).sort_values(ascending=False)
    
    print("Features ranked by their importance based on PCA loadings across retained components:")
    print(feature_importance)
    feature_importance.to_csv('PCA_components.csv')

    
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
    

# Function to process each subject
def process_subject(subject_data,):
    subject_id, admit_time = subject_data['SUBJECT_ID'].iloc[0], subject_data['admit_time'].iloc[0]

    subject_data = subject_data[(subject_data['CHARTTIME'] >= admit_time) & 
                                (subject_data['CHARTTIME'] <= admit_time + pd.Timedelta(hours=num_hours))]

    if not subject_data.empty:
        subject_features = set(subject_data['ITEMID'].map(id_conversion))
        return subject_id, subject_features
    
    return subject_id, None
    
# Filters chart data based on valid patients in the right time stamp
def filter_chart_data(chart_data, data, max_hours=48):
    chart_data['CHARTTIME'] = pd.to_datetime(chart_data['CHARTTIME'], errors='coerce')
    chart_data = chart_data.dropna(subset=['CHARTTIME'])

    admit_times = {row['SUBJECT_ID']: pd.to_datetime(row['ADMITTIME']) for _, row in data.iterrows()}
    chart_data = chart_data[chart_data['SUBJECT_ID'].isin(admit_times)]

    chart_data['admit_time'] = chart_data['SUBJECT_ID'].map(admit_times)
    chart_data = chart_data[chart_data['CHARTTIME'] <= (chart_data['admit_time'] + pd.Timedelta(hours=max_hours))]

    return chart_data

# Basic ML model testing
def basic_model(format_str = "%Y-%m-%d %H:%M:%S", impute = True,
                max_hours = 24):
    
    # Sets global variables to facilitate multiprocessing functions
    global id_conversion, num_hours
    num_hours = max_hours
    
    # Obtain discharge/death/admission times for label processing
    data = pd.read_csv('dataset/ADMISSIONS.csv')
    chart_data = pd.read_csv('dataset/CHARTEVENTS.csv',)
    #note_data = pd.read_csv('./dataset/NOTEEVENTS.csv')
    #prescription_data = pd.read_csv('dataset/PRESCRIPTIONS.csv')
    # Conversion dict for item IDs 
    conversion_data = pd.read_csv('dataset/D_ITEMS.csv')

    # Extract time-related discharge, death, and admit values and converts them to float
    data['ADMITTIME'] = pd.to_datetime(data['ADMITTIME'],)
    data['DISCHTIME'] = pd.to_datetime(data['DISCHTIME'])
    data['DEATHTIME'] = pd.to_datetime(data['DEATHTIME'],)

    admit_times = data['ADMITTIME']
    mapping_discharge = {data['SUBJECT_ID']: data['ADMITTIME'] for _, data in data.iterrows()}
    discharge_times = data['DISCHTIME']
    death_times = data['DEATHTIME']

    # Obtains labels
    death_labels = []
    hospital_stay = []
    for admit, dis, death in tqdm.tqdm(zip(admit_times, discharge_times, death_times), total = len(admit_times), desc = 'Processing mortality'):
       
        try:
            death_time = (death - admit).total_seconds()
            if math.isnan(death_time):
                raise ValueError
            hospital_stay.append(death_time)
            death_labels.append(1)
        except:
            hospital_stay.append((dis - admit).total_seconds())
            death_labels.append(0)          

    # Filter chart_data to include only entries within the specified time frame
    chart_data = filter_chart_data(chart_data, data, max_hours=max_hours)
    
    # Convert ITEMID to human-readable labels
    id_conversion = {row['ITEMID']: row['LABEL'] for _, row in conversion_data.iterrows()}
    
    # Convert CHARTTIME to datetime
    chart_data['CHARTTIME'] = pd.to_datetime(chart_data['CHARTTIME'])
    
    # Initialize structures for data storage
    all_features = set()
    feature_presence = {feature: [] for feature in id_conversion.values()}

    # Convert times outside the loop
    chart_data['CHARTTIME'] = pd.to_datetime(chart_data['CHARTTIME'])

    # Properly adds admit times to df
    chart_admit = [mapping_discharge[subject_id] if subject_id in mapping_discharge else None for subject_id in chart_data['SUBJECT_ID']]
    chart_data['admit_time'] = chart_admit
    chart_data['admit_time'] = pd.to_datetime(chart_data['admit_time'], errors = 'coerce')
    chart_data = chart_data.dropna(subset = ['admit_time'])

    # Ensure datetime format and handle NaN values
    chart_data['CHARTTIME'] = pd.to_datetime(chart_data['CHARTTIME'], errors='coerce')
    chart_data = chart_data.dropna(subset=['CHARTTIME'])

    # Use a dictionary for efficient feature tracking
    all_features = set()
    feature_presence = {}
    
    # Split data by subject ID
    grouped_data = [group for _, group in chart_data.groupby('SUBJECT_ID')]
    pool = Pool(processes = 16)

    # Filters for all patients that have chart data within the given timeframe
    results = []
    for result in tqdm.tqdm(pool.imap(process_subject, grouped_data), total=len(grouped_data), desc = 'Obtaining features'):
        if result[1] != None:
            results.append(result)

    # Aggregate results
    filtered_subjects = []
    for subject_id, subject_features in results:
        all_features = all_features.union(subject_features)
        feature_presence[subject_id] = {feature: (1 if feature in subject_features else 0) for feature in all_features}
        filtered_subjects.append(subject_id)

    feature_df = pd.DataFrame.from_dict(feature_presence, orient='index')

    feature_presence = {key: sum(feature_df[key])/len(feature_df[key]) for key in feature_df.columns if sum(feature_df[key])/len(feature_df[key]) > 0.04}
    print(f'{len(feature_presence)} Filtered Features')

    # Conversion for item IDs to features
    reverse_conversion = {v: k for k, v in id_conversion.items()}
    
    # Initialize a dictionary to keep track of the count of valid features per patient
    valid_feature_count = {subject_id: 0 for subject_id in data['SUBJECT_ID']}

    # Threshold for minimum number of features (10% of all features)
    min_feature_threshold = len(feature_presence) * 0.04

    # Iterate through each subject ID and obtain feature values
    patient_data = {}

    # Convert subject_ids to set for faster membership testing
    filtered_subjects_set = set(filtered_subjects)

    # Prepare chart_data by setting index and sorting it for faster access
    chart_data = chart_data.set_index(['SUBJECT_ID', 'ITEMID']).sort_index()

    # Initialize patient_data and valid_feature_count dictionaries
    patient_data = {}
    valid_feature_count = {subject_id: 0 for subject_id in filtered_subjects_set}

    # Convert feature_presence to a dictionary for faster access
    feature_to_item = {feature: reverse_conversion.get(feature) for feature in feature_presence}

    mortality_labels = []
    for num, subject_id in tqdm.tqdm(enumerate(set(data['SUBJECT_ID'])), desc='Obtaining Patient Data', total = len(data)):
        if subject_id in filtered_subjects_set:
            features = {}

            for feature, item_id in feature_to_item.items():
                if item_id is not None:
                    # Access the subset of chart_data efficiently
                    feature_values = chart_data.loc[(subject_id, item_id), 'VALUE'] if (subject_id, item_id) in chart_data.index else pd.Series(dtype='float64')

                    # Drop NaN values and reset index to avoid issues with iloc
                    feature_values = feature_values.dropna().reset_index(drop=True)

                    # Check if the resulting Series is empty to avoid IndexError
                    feature_value = feature_values.iloc[0] if not feature_values.empty else None

                    if feature_value is not None:
                        try:
                            feature_value = float(feature_value)
                            valid_feature_count[subject_id] += 1
                        except ValueError:
                            feature_value = None
                    features[feature] = feature_value

            if valid_feature_count[subject_id] >= min_feature_threshold:
                patient_data[subject_id] = features
                mortality_labels.append(death_labels[num])

    data = mortality_labels    
    patient_data = pd.DataFrame.from_dict(patient_data, orient='index')
    patient_data = patient_data.dropna(axis=1, how='all')

    # Number of filtered patients
    print(f'{len(patient_data.columns)} features remaining after final filtration')
    print(f'{len(patient_data)} patients remaining after final filtration')
    
    labellers = {}
    string_columns_count = 0  # Counter for columns with string values
    for column in patient_data.columns:

        # Check if there are any string values in the column
        if patient_data[column].apply(lambda x: isinstance(x, str)).any():

            try:
                # Increment the counter as this column contains string data
                string_columns_count += 1

                # Fill NaN values with a placeholder
                patient_data[column] = patient_data[column].fillna('missing')

                # Initialize LabelEncoder and transform values
                le = LabelEncoder()
                patient_data[column] = le.fit_transform(patient_data[column])
                labellers[column] = le

                # Identify the encoded value for 'missing' placeholder
                missing_label = le.transform(['missing'])[0]

                # Replace the 'missing' encoded value with NaN in the dataframe
                patient_data[column] = patient_data[column].replace(missing_label, None)
            except:
                patient_data[column] = patient_data[column].apply(lambda x: None if isinstance(x, str) else x)
                
    # Removes "Time" feature
    patient_data = patient_data.drop(columns = ["Time"])
    
    features = patient_data        
    if impute == True:
        scaler = StandardScaler()
        imputer = MICE()
        numpy_features = np.array([[item for item in row] for row in features.to_numpy()])
        numpy_features = scaler.fit_transform(numpy_features)
        imputed_data = imputer.fit_transform(numpy_features)
        features = pd.DataFrame(imputed_data, columns=features.columns, index=features.index)
    
    # Analyzes feature importance via a variety of methods
    analyze_features(features_scaled = features.to_numpy(), feature_names = features.columns, y = data)
    
    data = pd.concat((features, pd.DataFrame({"Death":data})), axis = 1)
    if impute == False:
        data = data.dropna()
        
    data = equalize_data(data) 
    data = data.sample(frac = 1) 
    print(f'Processed data length: {len(data)}')

    # Data prep
    print(data)
    train_data, test_data = train_test_split(data, test_size = 0.2, random_state=42)
    X_train, y_train = train_data.iloc[:, :-1].to_numpy(), train_data.iloc[:, -1].to_numpy()
    X_test, y_test = test_data.iloc[:, :-1].to_numpy(), test_data.iloc[:, -1].to_numpy()
    
    scaler = StandardScaler()
    print(X_train)
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
    plt.plot(fpr, tpr, label = f'Random Forest (AUC: {round(auc, 3)})', color = 'red')
    plt.plot([0, 1], [0, 1], label = f'Random \n (AUC: {.500}', linestyle = '--', color = 'blue')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc = "lower right")
    plt.title('Random Forest Classifier on Predicting Mortality based on Hospital Stay')
    plt.savefig('RF_AUC.png')
  
  
# Main runtime
if __name__ == '__main__':
    basic_model()
    