import pandas as pd
import joblib
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv("dados/Ortopedia_Coluna.csv", delimiter=';')

#Isolar a base de dados com a classe minoritaria
minoritaria = data[data['Fusao_de_Vertebras']==1]
majoritaria = data[data['Fusao_de_Vertebras']==0]
minoritaria_upsample = resample(minoritaria, replace=True, n_samples=7900, random_state=0)

data_balanceado = pd.concat([majoritaria, minoritaria_upsample])

attributes = data_balanceado.drop('Fusao_de_Vertebras', axis=1)
classes = data_balanceado['Fusao_de_Vertebras']

new_attributes = pd.get_dummies(attributes);

print(new_attributes)

X_train, X_test, y_train, y_test = train_test_split(new_attributes, classes, test_size=0.20)

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Resultado da Avaliação do Modelo")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(classifier, 'models/ortopedia.joblib')
classifier = joblib.load('models/ortopedia.joblib')

print("Predict")
print(classifier.predict(X_test))
print("Predict proba")
print(classifier.predict_proba(X_test))

