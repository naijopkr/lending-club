import pandas as pd
import random
import joblib

from tensorflow.keras.models import load_model

model = load_model('lending-club.h5')
scaler = joblib.load('data/scaler.save')

df = pd.read_csv('data/cleansed_data.csv')
random_index = random.randint(0, len(df))

customer = scaler.transform(
    [df.drop(['loan_repaid', 'loan_status'], axis=1).loc[random_index]]
)
customer_status = df.loc[random_index]['loan_repaid']

y_prob = model.predict(customer)

response = 1 if y_prob > .5 else 0

print(response == customer_status)
