
import config
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn import svm
import joblib

root = config.ROOT_PATH
dirData = 'Dados\\'
dirModel = 'Models\\'
fileModel = 'Diabetes_Classifier.joblib'

def setModel(fileIn, separador=',', blocks=30):
    ### Normalizar o arquivo
    path = dirData
    df = pd.read_csv(path + fileIn, sep=separador)

    # Divide os dados em dois conjuntos: Atributos e Classes
    attributes = df.drop('class', axis=1)
    classes = df['class']

    X_train, X_test, y_train, y_test = train_test_split(attributes, classes, test_size=0.20)        
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)

    #Aplicar o modelo gerado sobre os dados separados para testes
    y_pred = classifier.predict(X_test)

    fileName = dirModel + fileModel
        
    #Salvar o modelo para uso posterior
    joblib.dump(classifier, fileName)               
        
    validateModel(classifier, X_test, y_test, y_pred, 1, blocks)
    print('\n')
    validateModel(classifier, X_test, y_test, y_pred, 2, blocks)

def testModel(fileIn, separator, blocks):

    path = dirData
    df = pd.read_csv(path + fileIn, sep=separator)

    # Divide os dados em dois conjuntos: Atributos e Classes
    attributes = df.drop('class', axis=1)
    classes = df['class']

    X_train, X_test, y_train, y_test = train_test_split(attributes, classes, test_size=0.20)  

    classifier = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    return None
    
def validateModel(classifier, X_test, y_test, y_pred, type, blocks):

    if (type == 1):
        scores_1 = cross_val_score(classifier, X_test, y_test, cv=blocks)    
        print('cross_val_score')
        print(scores_1)
        print('Precisão média:', scores_1.mean(), '\n')
    elif(type == 2):
        scores_2 = cross_validate(classifier, X_test, y_test, cv=blocks)
        print('cross_validate')
        print(scores_2)
        print('Precisão média:', scores_2['test_score'].mean(), '\n')

    #Avaliar o modelo: Acurácia e matriz de contingência
    print("Resultado da Avaliação do Modelo")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

setModel('Diabetes.csv')

