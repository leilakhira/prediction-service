from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import os
import time
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
import warnings
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
# Ignorer un avertissement spécifique (par exemple, le FutureWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, this is your microservice!'

IP_table = pd.read_csv("IpAddress_to_Country.csv")   # Country from IP information
def times_to_delta_sec(row):
    purchase_sec = pd.to_datetime(row.purchase_time, format='%Y-%m-%d %H:%M:%S')
    signup_sec = pd.to_datetime(row.signup_time, format='%Y-%m-%d %H:%M:%S')
    
    return (purchase_sec - signup_sec).seconds
# function that takes an IP address as argument and returns country associated based on IP_table

def IP_to_country(ip) :
    try :
        return IP_table.country[(IP_table.lower_bound_ip_address < ip)                            
                                & 
                                (IP_table.upper_bound_ip_address > ip)].iloc[0]
    except IndexError :
        return "Unknown"
@app.route('/process_data', methods=['POST'])
def process_data():
    try:
        # Vérifier si un fichier CSV est présent dans la requête
        if 'file' not in request.files:
            return jsonify({"error": "No CSV file provided."})

        file = request.files['file']
    except Exception as e:
        return jsonify({"error": str(e)})
    dataset = pd.read_csv(file)              # Users information
    print("[Country mapping]")  
    # To affect a country to each IP :
    IP_table.upper_bound_ip_address.astype("float")
    IP_table.lower_bound_ip_address.astype("float")
    dataset.ip_address.astype("float")
    dataset["IP_country"] = dataset.ip_address.apply(IP_to_country)

    # Since this code is time consuming to run, we have saved the result in a file with the following line of code :
    dataset.to_csv("Fraud_data_with_country.csv")
    features = ['user_id', 'signup_time', 'purchase_time', 'purchase_value', 'device_id', 'source', 'browser', 'sex', 'age', 'IP_country', 'class']
    dataset = pd.read_csv('Fraud_data_with_country.csv', usecols=features)[features]
    print("[Time delta]")
    dataset['time_delta_sec']=dataset.apply(times_to_delta_sec, axis=1)
    print("[Device frequency]")

    device_freq = dataset.groupby('device_id').count().user_id.reset_index(name='device_freq')
    device_freq.to_csv('Device_Frequency.csv')
    device_freq = pd.read_csv('Device_Frequency.csv', usecols=['device_id', 'device_freq'])


    dataset = dataset.join(device_freq.set_index('device_id'), on='device_id')

    dataset.to_csv('Fraud_data_with_country_freq_time.csv')
    country_ratio = dataset[['IP_country', 'class']].groupby('IP_country').mean().reset_index(names='country').rename(columns={'class':'ratio'})
    country_ratio.to_csv('Country_Fraud_Ratio.csv')
    country_ratio = pd.read_csv('Country_Fraud_Ratio.csv', usecols=['country', 'ratio'])
    dataset = dataset.join(country_ratio.set_index('country'), on='IP_country')
    dataset.to_csv('Fraud_data_with_country_freq_time.csv')

    print("[Encoding]")

    features = ['purchase_value', 'source', 'browser', 'sex', 'age', 'ratio', 'device_freq', 'time_delta_sec', 'class']

    dataset_raw = pd.read_csv('Fraud_data_with_country_freq_time.csv', usecols=features)[features]
    features_to_encode = ['source', 'browser', 'sex']
    dataset_encoded = pd.get_dummies(dataset_raw, columns=features_to_encode, dtype=int)

    print("[Normalization]")

    X_not_std = dataset_encoded[['purchase_value', 'age', 'device_freq', 'time_delta_sec', 'ratio']]

    scaler = RobustScaler()
    X_std = scaler.fit_transform(X_not_std)
    dataset_encoded[['purchase_value', 'age', 'device_freq', 'time_delta_sec', 'ratio']] = X_std
    dataset_encoded_= dataset_encoded

   #modelling 
    dataset_encoded_normalized = pd.read_csv("dataset_encoded_normalized_process.csv") 
    y = dataset_encoded_normalized['class']
    X = dataset_encoded_normalized.drop(['class'], axis=1)
    #split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
    
    gbm_tuned = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,max_depth=6, min_samples_split = 500, min_samples_leaf = 60, max_features = 16,subsample=0.9, random_state=41)
    gbm_tuned.fit(X_train, y_train)
    # predict class labels 0/1 for the test set
    predicted = gbm_tuned.predict(X_test)
    # generate class probabilities
    probs = gbm_tuned.predict_proba(X_test)
    # Generate evaluation metrics
    accuracy = accuracy_score(y_test, predicted)
    roc_auc = roc_auc_score(y_test, probs[:, 1])
    f1 = f1_score(y_test, predicted)
    cm = confusion_matrix(y_test, predicted)
    recall = float(cm[1, 1]) / (cm[1, 0] + cm[1, 1])
    precision = float(cm[1, 1]) / (cm[1, 1] + cm[0, 1])
    cm_df=pd.DataFrame(cm, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])

    # Create and return a dictionary with the evaluation metrics
    evaluation_results = {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "f1_score": f1,
       "confusion_matrix": cm_df.to_dict(),
        "recall": recall,
        "precision": precision
    }
    Z = dataset_encoded_.drop(['class'], axis=1)
    predicted_ = gbm_tuned.predict(X)
    result=pd.DataFrame(predicted_)
    return evaluation_results,result

if __name__ == '__main__':
    app.run(debug=True, port=8000)

