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

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, this is your microservice!'



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


        
if __name__ == '__main__':
    app.run(debug=True, port=8002)

