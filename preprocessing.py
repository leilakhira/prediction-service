import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split



def times_to_delta_sec(row):
    purchase_sec = pd.to_datetime(row.purchase_time, format='%Y-%m-%d %H:%M:%S')
    signup_sec = pd.to_datetime(row.signup_time, format='%Y-%m-%d %H:%M:%S')
    
    return (purchase_sec - signup_sec).seconds




def preprocess(new_data):

    dataset=pd.read_csv("data/Fraud_Data.csv")

    print("Time delta...", end="")
    dataset['time_delta_sec'] = dataset.apply(times_to_delta_sec, axis=1)
    new_data['time_delta_sec'] = new_data.apply(times_to_delta_sec, axis=1)
    print("DONE.")



    # print("Device frequency...", end="")
    # device_freq = pd.read_csv('data/Device_Frequency.csv', usecols=['device_id', 'device_freq'])
    # dataset = dataset.join(device_freq.set_index('device_id'), on='device_id')
    # print("DONE.")




    #dataset.drop(columns=['user_id', 'signup_time', 'purchase_time', 'device_id'], inplace=True)

    X = dataset.drop(columns=['class'])
    y = dataset['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    new_data.reset_index(drop=True, inplace=True)




    print("Normalization...", end="")
    scaler = RobustScaler()
    scaler.fit(X_train[['purchase_value', 'age', 'time_delta_sec', 'ip_address']])

    scaled = scaler.transform(new_data[['purchase_value', 'age', 'time_delta_sec', 'ip_address']])
    new_data[['purchase_value', 'age', 'time_delta_sec', 'ip_address']] = scaled
    print("DONE.")



    print("Encoding...", end="")
    features_to_encode = np.stack([X_train['source'], X_train['browser'], X_train['sex']], axis=1).reshape((-1, 3))
    encoder = OneHotEncoder(sparse_output=False).fit(features_to_encode)
    
    encoded = encoder.transform(np.stack([new_data['source'], new_data['browser'], new_data['sex']], axis=1).reshape((-1, 3)))

    encoded_cols = np.array([])
    for i in (encoder.categories_):
        encoded_cols = np.concatenate((encoded_cols,i), axis=0)

    encoded = pd.DataFrame(encoded, columns=encoded_cols, dtype=int)
    new_data = pd.concat([new_data, encoded], axis=1).drop(columns=['source', 'browser', 'sex'])
    print("DONE.")


    return new_data