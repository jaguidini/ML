import config
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn import svm
import joblib

class Executar():
    ### Construtor para a classe, inicializando as variáveis locais
    def __init__(self):        
        self.__root = config.ROOT_PATH
        self.__dirData = 'Dados\\'
        self.__dirModel = 'Models\\'
        self.__fileModel = 'decisiontree1.joblib'

    def setModel(self, fileIn, fileOut, separador=','):
        ### Normalizar o arquivo
        path = self.__dirData
        df = pd.read_csv(path + fileIn, sep=separador)

        # Divide os dados em dois conjuntos: Atributos e Classes
        attributes = df.drop('class', axis=1)
        classes = df['class']

        X_train, X_test, y_train, y_test = train_test_split(attributes, classes, test_size=0.20)        
        classifier = DecisionTreeClassifier()
        classifier.fit(X_train, y_train)

        #Aplicar o modelo gerado sobre os dados separados para testes
        y_pred = classifier.predict(X_test)
        #print(y_pred)

        #Avaliar o modelo: Acurácia e matriz de contingência
        print("Resultado da Avaliação do Modelo")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        
        #Classificar uma nova instância
        print("Classificar [1,103,30,38,83,43.3,0.183,33] => tested_negative")
        nova_instancia=[[1,103,30,38,83,43.3,0.183,33]]
        print(classifier.predict(nova_instancia))

        fileName = self.__dirModel + self.__fileModel
        #Salvar o modelo para uso posterior
        joblib.dump(classifier, fileName)

        classifier = joblib.load(fileName)
        nova_instancia=[[0,137,40,35,168,43.1,2.288,33]]
        print("Com o modelo salvo: ")
        print("Classificar [0,137,40,35,168,43.1,2.288,33] => tested_positive")
        print(classifier.predict(nova_instancia))


        #df_normal = pd.get_dummies(df)
        #filename = path + fileOut
        #df_normal.to_csv(filename, sep=';', index = None, header=True)

        #print(df_normal)

        #### Gerar o modelo
        #path = self.__root + self.__dirModel
        #filename = path + self.__fileModels
        #kmeans = KMeans(n_clusters=clusters).fit(df_normal)        
        #pickle.dump(kmeans, open(filename, 'wb'))
        
        #return 'Modelo {} gerado com {} clusters.'.format(self.__fileKMeansModel), 200

    def testModel(self, fileIn, separator, blocks):

        path = self.__dirData
        df = pd.read_csv(path + fileIn, sep=separator)

        # Divide os dados em dois conjuntos: Atributos e Classes
        attributes = df.drop('class', axis=1)
        classes = df['class']

        X_train, X_test, y_train, y_test = train_test_split(attributes, classes, test_size=0.20)  

        classifier = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
        return None
    
    def validateModel(self, classifier, X_test, y_test, type, blocks):

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

        #Especificidade
        y_pred = classifier.predict(X_test)
        #print(y_pred)

        #Avaliar o modelo: Acurácia e matriz de contingência
        print("Resultado da Avaliação do Modelo")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        y_pred = classifier.predict_proba(X_test)
        print(y_pred)
        