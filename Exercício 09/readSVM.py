import joblib

kernel = "linear"
classifier = joblib.load('models/svm_{0}.joblib'.format(kernel))
param=[[7.7,0.49,0.26,1.9,0.062,9,31,0.9966,3.39,0.64,9.6],[8,0.59,0.16,1.8,0.065,3,16,0.9962,3.42,0.92,10.5]]
print("Predict")
print(classifier.predict(param))
print("Predict proba")
print(classifier.predict_proba(param))
