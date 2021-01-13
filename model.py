import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("E:/Diabtes/diabetes.csv")

df = df.rename(columns={'DiabetesPedigreeFunction':'DPF'})
# print(df)
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
# print(df)
df['Glucose'] = df['Glucose'].fillna(df['Glucose'].mean())
df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].mean())
df['SkinThickness'] = df['SkinThickness'].fillna(df['SkinThickness'].mean())
df['Insulin'] = df['Insulin'].fillna(df['Insulin'].mean())
df['BMI'] = df['BMI'].fillna(df['BMI'].mean())

target = df['Outcome']
feature = df.drop('Outcome', axis=1)

# print(feature)


X_train, X_test, y_train, y_test = train_test_split(feature.values, target.values, test_size=0.30, random_state=42)

# print(X_test)
model = XGBClassifier()
model.fit(X_train,y_train)

#y_pred = model.predict(X_test)

#accuracy = accuracy_score(y_test, y_pred)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))

filename = 'prediction.pkl'
pickle.dump(model, open(filename, 'wb'))
# print(feature.columns)
