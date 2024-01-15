import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/Fraud_Data.csv")

# data = preprocessing.preprocess(data)
# print(data.isna().sum())

X = data.drop(columns=['class'])
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

X_train = preprocessing.preprocess(X_train)
X_test = preprocessing.preprocess(X_test)

print(X_train.isna().sum())
print(X_test.isna().sum())