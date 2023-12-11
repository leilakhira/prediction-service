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
import pickle

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, this is your microservice!'


############################################## Data Processing  ###############################

def times_to_delta_sec(row):
    purchase_sec = pd.to_datetime(row.purchase_time, format='%Y-%m-%d %H:%M:%S')
    signup_sec = pd.to_datetime(row.signup_time, format='%Y-%m-%d %H:%M:%S')
    
    return (purchase_sec - signup_sec).seconds

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

    print("[Time delta]")
    dataset['time_delta_sec'] = dataset.apply(times_to_delta_sec, axis=1)


    print("[Device frequency]")
    # device_freq = dataset.groupby('device_id').count().user_id.reset_index(name='device_freq')
    # device_freq.to_csv('data/Device_Frequency.csv')
    device_freq = pd.read_csv('data/Device_Frequency.csv', usecols=['device_id', 'device_freq'])
    dataset = dataset.join(device_freq.set_index('device_id'), on='device_id')


    print("[Encoding]")
    features_to_encode = ['source', 'browser', 'sex']
    dataset = pd.get_dummies(dataset, columns=features_to_encode, dtype=int)



    print("[Normalization]")
    X_not_std = dataset[['purchase_value', 'age', 'device_freq', 'time_delta_sec', 'ip_address']]
    scaler = RobustScaler()
    X_std = scaler.fit_transform(X_not_std)
    dataset[['purchase_value', 'age', 'device_freq', 'time_delta_sec', 'ip_address']] = X_std


    dataset.drop(columns=['user_id', 'signup_time', 'purchase_time', 'device_id'], inplace=True)
   # dataset.to_csv('data/dataset_encoded_normalized.csv')

    result={
        "result":dataset.to_dict(orient="records") 
    }      
   
    return result
    
##return dataset_encoded_

################################ Modelling ###############################

@app.route('/train_model', methods=['GET'])
def train_model():
    try:
        dataset_encoded_normalized = pd.read_csv("data/dataset_encoded_normalized.csv") 
        dataset_encoded_normalized.drop(['Unnamed: 0'], axis=1, inplace=True)
       # print(dataset_encoded_normalized)
        y = dataset_encoded_normalized['class']
        X = dataset_encoded_normalized.drop(['class'], axis=1)
        #split into train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
        
        gbm_tuned = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,max_depth=6, min_samples_split = 500, min_samples_leaf = 60, max_features = 16,subsample=0.9, random_state=41)
        gbm_tuned.fit(X_train, y_train)
        # Serialize and save the trained model using pickle
        with open('trained_model.pkl', 'wb') as model_file:
            pickle.dump(gbm_tuned, model_file)

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
        return evaluation_results
    except Exception as e:
           return jsonify({'error': str(e)})


############################################## Prediction ###############################

@app.route('/prediction_model', methods=['POST'])
def prediction_model():
    try:
        # Appeler la fonction process_data pour obtenir la réponse JSON
        response = process_data()

        # Assurez-vous que la réponse contient la clé 'result'
        if 'result' not in response:
            return jsonify({'error': 'Invalid response format. Missing "result" key.'})

        # Extraire la DataFrame du dictionnaire de réponse
        dataset_encoded_normalized = pd.DataFrame(response['result'])
       # dataset_encoded = pd.read_csv("data/dataset_encoded_normalized_process.csv") 
       #  Z = dataset_encoded.drop(['class'], axis=1)
        X = dataset_encoded_normalized.drop(['class'], axis=1)

        # Charger le modèle entraîné depuis le fichier
        with open('trained_model.pkl', 'rb') as model_file:
            gbm_tuned = pickle.load(model_file)

        # predict class labels 0/1 for the test set
        predicted = gbm_tuned.predict(X)
        # Créer un DataFrame avec les prédictions
        predictions_df = pd.DataFrame({'predictions': predicted})

        # Enregistrer le DataFrame dans un fichier Excel
        predictions_df.to_csv('data/predictions_.csv', index=False)
        
        # Retourner les prédictions
        return jsonify(predicted.tolist())
    
    except Exception as e:
           return jsonify({'error': str(e)})

        
if __name__ == '__main__':
    app.run(debug=True, port=8000)

