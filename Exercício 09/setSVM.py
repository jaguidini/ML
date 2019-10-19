import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def get_wine_class(value):
    if value <= 3:
        return 'C'
    if value <= 6:
        return 'B'
    return 'A'

dfRed = pd.read_csv("dados/winequality-red.csv", delimiter=';')
dfRed['quality'] = dfRed['quality'].map(get_wine_class)

#Isolar a base de dados com a classe minoritaria
C = dfRed[dfRed['quality']=='C']
B = dfRed[dfRed['quality']=='B']
A = dfRed[dfRed['quality']=='A']

#print(dfRed['quality'].value_counts())

upsample_C = resample(C, replace=True, n_samples=1372, random_state=0)
upsample_B = resample(A, replace=True, n_samples=1372, random_state=0)

dfBalance = pd.concat([upsample_C, B, upsample_B])

attributes = dfBalance.drop('quality', axis=1)
classes = dfBalance['quality']

X_train, X_test, y_train, y_test = train_test_split(attributes, classes, test_size=0.20)
### pode ser rbf, poly, linear
kernel = "linear"
classifier = svm.SVC(kernel = kernel, C = 1, probability=True, gamma='auto');
model = classifier.fit(X_train, y_train) 
retorno = model.predict(X_test)

# Acurácia e matriz de contingência
print("Resultado da Avaliação do Modelo")
print(confusion_matrix(y_test, retorno))
print(classification_report(y_test, retorno))


joblib.dump(classifier, 'models/svm_{0}.joblib'.format(kernel))

