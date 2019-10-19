import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

diabetes = pd.read_csv("dados/diabetes.csv", delimiter=',')

attributes = diabetes.drop('class', axis=1)
classes = diabetes['class']

new_attributes = pd.get_dummies(attributes);

X_train, X_test, y_train, y_test = train_test_split(new_attributes, classes, test_size=0.30)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Resultado da Avaliação do Modelo Logistica")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("Classificar [6,25,46,59,0,46.2,1.1,21]")

param=[[6,130,46,59,0,46.2,1.1,21]]
print(classifier.predict(param))

joblib.dump(classifier, 'models/logistica.joblib')
