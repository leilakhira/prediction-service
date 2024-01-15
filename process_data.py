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
import preprocessing
import logging

logging.basicConfig(level=logging.DEBUG, filename='process_data.log')

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
        logging.debug("%s", e)
        return jsonify({"error": str(e)})
    
    dataset = pd.read_csv(file)              # Users information

    preprocessed_data = preprocessing.preprocess(dataset)

    result={
        "result":preprocessed_data.to_dict(orient="records") 
    }      
   
    return result
    

        
if __name__ == '__main__':
    app.run(debug=True, port=8001)

