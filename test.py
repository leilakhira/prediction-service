import requests

url = "http://localhost:8000/prediction_model"  # Assurez-vous de changer l'URL en fonction de votre configuration

# Charger votre dataset CSV
with open("data/Fraud_Data.csv", "rb") as file:
    files = {'file': ('data/Fraud_Data.csv', file)}
    response = requests.post(url, files=files)

print(response.json())
