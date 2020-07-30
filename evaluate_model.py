import pandas as pd
import joblib

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model


model = load_model('lending-club.h5')
model_loss = pd.read_csv('data/model_loss.csv')
test_data = pd.read_csv('data/test_data.csv')
scaler = joblib.load('data/scaler.save')


X_test = scaler.transform(
    test_data.drop('loan_repaid', axis=1)
)
y_test = test_data['loan_repaid']


y_pred = model.predict_classes(X_test)


confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
