# -*- coding: utf-8 -*-
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

diabetes = pd.read_csv("dados/diabetes.csv", delimiter=',')

attributes = diabetes.drop('class', axis=1)
classes = diabetes['class']

new_attributes = pd.get_dummies(attributes);

X_train, X_test, y_train, y_test = train_test_split(new_attributes, classes, test_size=0.30)

classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Resultado da Avaliação do Modelo Random forest classifier")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("Classificar [6,25,46,59,0,46.2,1.1,21]")
param=[[6,130,46,59,0,46.2,1.1,21]]
print(classifier.predict(param))

#Salvar o modelo para uso posterior
joblib.dump(classifier, 'models/random_forest.joblib')
