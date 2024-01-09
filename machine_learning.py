from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
import os
import time
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
import warnings
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
# Ignorer un avertissement spécifique (par exemple, le FutureWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
import pickle
import preprocessing

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, this is your microservice!'


############################################## Data Processing  ###############################

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

    preprocessed_data = preprocessing.preprocess(dataset)

    result={
        "result":preprocessed_data.to_dict(orient="records") 
    }      
   
    return result
    
##return dataset_encoded_

################################ Modelling ###############################

@app.route('/train_model', methods=['GET'])
def train_model():
    try:
        dataset=pd.read_csv("data/Fraud_Data.csv")
        dataset.drop(columns=['user_id', 'signup_time', 'purchase_time', 'device_id'], inplace=True)
        X = dataset.drop(columns=['class'])
        y = dataset['class']

        #split into train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)
        X_train = preprocessing.preprocess(X_train)
        X_test = preprocessing.preprocess(X_test)

        dt = DecisionTreeClassifier(ccp_alpha=0, criterion='gini', max_depth=8, max_features=None, min_samples_leaf=2, min_samples_split=2, splitter='random')
        smote = SMOTE() # default : 5 nearest neighbors
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        dt.fit(X_train_smote, y_train_smote)
        
        # Serialize and save the trained model using pickle
        with open('trained_model.pkl', 'wb') as model_file:
            pickle.dump(dt, model_file)

        # predict class labels 0/1 for the test set
        predicted = dt.predict(X_test)
        # generate class probabilities
        probs = dt.predict_proba(X_test)
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

