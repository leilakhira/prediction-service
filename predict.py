from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
import warnings
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix
# Ignorer un avertissement spécifique (par exemple, le FutureWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
import pickle
import requests
import preprocessing

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, this is your microservice!'


############################################## Prediction ###############################

url_process = "http://127.0.0.1:8001/process_data"

@app.route('/prediction_model', methods=['POST'])
def prediction_model():
    try:
        # Appeler la fonction process_data pour obtenir la réponse JSON
        response = requests.post(url_process, data=request.files['file'])

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
            dt = pickle.load(model_file)

        # predict class labels 0/1 for the test set
        predicted = dt.predict(X)
        # Créer un DataFrame avec les prédictions
        predictions_df = pd.DataFrame({'predictions': predicted})

        # Enregistrer le DataFrame dans un fichier Excel
        predictions_df.to_csv('data/predictions_.csv', index=False)
        
        # Retourner les prédictions
        return jsonify(predicted.tolist())
    
    except Exception as e:
           return jsonify({'error': str(e)})

        
if __name__ == '__main__':
    app.run(debug=True, port=8003)

