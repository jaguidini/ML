import joblib

classifier = joblib.load('models/diabetes.joblib')
param=[[0,100,40,35,168,43.1,2.288,33]]
print("Predict")
print(classifier.predict(param))
print("Predict proba")
print(classifier.predict_proba(param))