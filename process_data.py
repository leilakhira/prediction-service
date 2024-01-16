from flask import Flask, request, jsonify
import pandas as pd
import warnings
import preprocessing
import traceback
import logging

# Ignorer un avertissement spécifique (par exemple, le FutureWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

logging.basicConfig(level=logging.DEBUG, filename='process_data.log')

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, this is the preprocessing microservice!'


############################################## Data Processing  ###############################

@app.route('/process_data', methods=['POST'])
def process_data():
    logging.info("Start of preprocessing")

    try:
        # Vérifier si un fichier CSV est présent dans la requête
        if 'file' not in request.files:
            logging.error("Data missing in request")
            return jsonify({"error": "No CSV file provided."})

        file = request.files['file']
        dataset = pd.read_csv(file)  # User information
        preprocessed_data = preprocessing.preprocess(dataset)

        result={
            "result":preprocessed_data.to_dict(orient="records")
        }


        logging.info("End of preprocessing")
        return result


    except Exception as e:
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e)})
   
    
        
if __name__ == '__main__':
    app.run(debug=True, port=8001)

