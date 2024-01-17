from flask import Flask, request, jsonify
import pandas as pd
import warnings
import pickle
import logging
import traceback
import preprocessing

# Ignorer un avertissement spécifique (par exemple, le FutureWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

logging.basicConfig(level=logging.DEBUG, filename='log/predict.log')

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, this is the prediction microservice!'


############################################## Prediction ###############################


@app.route('/prediction_model', methods=['POST'])
def prediction_model():
    logging.info("Start of prediction")

    try:
        file = request.files['file']
        data = pd.read_csv(file)

        X = data.drop(['class'], axis=1)
        X = preprocessing.preprocess(X)

        # Charger le modèle entraîné depuis le fichier
        with open('model/trained_model.pkl', 'rb') as model_file:
            dt = pickle.load(model_file)

        # predict class labels 0/1 for the test set
        predicted = dt.predict(X)
        # Créer un DataFrame avec les prédictions
        predictions_df = pd.DataFrame({'predictions': predicted})

        # Enregistrer le DataFrame dans un fichier Excel
        predictions_df.to_csv('data/predictions_.csv', index=False)
        
        # Retourner les prédictions
        logging.info("End of prediction")
        return jsonify(predicted.tolist())
    
    except Exception as e:
           logging.error(traceback.format_exc())
           return jsonify({'error': str(e)})

        
if __name__ == '__main__':
    app.run(debug=True, port=8003)

