import requests


train_url = "http://localhost:8002/train_model"  # Assurez-vous de changer l'URL en fonction de votre configuration


response = requests.get(train_url)

print(response.json())