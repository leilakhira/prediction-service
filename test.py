import requests


pred_url = "http://localhost:8003/prediction_model"  # Assurez-vous de changer l'URL en fonction de votre configuration
process_url = "http://localhost:8001/process_data"

# Charger votre dataset CSV
with open("data/Fraud_Data.csv", "rb") as file:
    files = {'file': ('data/Fraud_Data.csv', file)}
    response = requests.post(pred_url, files=files)


print(response.json())