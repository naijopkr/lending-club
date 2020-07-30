import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

df = pd.read_csv('data/cleansed_data.csv')
df.head()

X = df.drop(['loan_status', 'loan_repaid'], axis=1)
y = df['loan_repaid']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=101
)

test_data = pd.concat([X_test, y_test], axis=1)
test_data.to_csv('data/test_data.csv', index=False)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, 'data/scaler.save')

model = Sequential()

model.add(Dense(78, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(39, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(19, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(
    x=X_train,
    y=y_train,
    epochs=25,
    batch_size=256,
    validation_data=(X_test, y_test),
    verbose=1
)

model.save('lending-club.h5')

model_loss = pd.DataFrame(model.history.history)
model_loss.to_csv('data/model_loss.csv', index=False)
