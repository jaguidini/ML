import config
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn import svm
from sklearn.utils import resample
import joblib

class Executar():
    ### Construtor para a classe, inicializando as variáveis locais
    def __init__(self):        
        self.__root = config.ROOT_PATH
        self.__dirData = 'Dados\\'
        self.__dirModel = 'Models\\'
        self.__fileModel = 'Ortopedia_Model.joblib'        
        self.__fileNormal = 'Ortopedia_Coluna_Normal.csv'

    def setModel(self, fileIn, separador=',', blocks=30):
        ### Normalizar o arquivo
        path = self.__dirData
        df = pd.read_csv(path + fileIn, sep=separador)

        # Isolar a base de dados com a classe minoritária
        minoritario = df[df['Fusao_de_Vertebras']==1]
        majoritario = df[df['Fusao_de_Vertebras']==0]
        minoritario_upsamples = resample(minoritario, replace=True, n_samples=7900, random_state=123)
        df_balanceado = pd.concat([majoritario, minoritario_upsamples])

        # Normalizar
        df_normal = pd.get_dummies(df_balanceado)
        #filename = path + self.__fileNormal
        #df_normal.to_csv(filename, sep=';', index = None, header=True)

        # Divide os dados em dois conjuntos: Atributos e Classes
        attributes = df_normal.drop('Fusao_de_Vertebras', axis=1)
        classes = df_normal['Fusao_de_Vertebras']
        
        X_train, X_test, y_train, y_test = train_test_split(attributes, classes, test_size=0.20)        
        classifier = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
        #classifier = DecisionTreeClassifier()
        #classifier.fit(X_train, y_train)

        #Aplicar o modelo gerado sobre os dados separados para testes
        y_pred = classifier.predict(X_test)

        #Avaliar o modelo: Acurácia e matriz de contingência
        print("Resultado da Avaliação do Modelo")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        fileName = self.__dirModel + self.__fileModel

        #Salvar o modelo para uso posterior
        joblib.dump(classifier, fileName)               
        
        self.validateModel(classifier, X_test, y_test, y_pred, 1, blocks)
        print('\n')
        self.validateModel(classifier, X_test, y_test, y_pred, 2, blocks)


        #Classificar uma nova instância
        #print("Classificar [1,103,30,38,83,43.3,0.183,33] => tested_negative")
        #nova_instancia=[[1,103,30,38,83,43.3,0.183,33]]
        #print(classifier.predict(nova_instancia))

        #classifier = joblib.load(fileName)
        #nova_instancia=[[0,137,40,35,168,43.1,2.288,33]]
        #print("Com o modelo salvo: ")
        #print("Classificar [0,137,40,35,168,43.1,2.288,33] => tested_positive")
        #print(classifier.predict(nova_instancia))

    def execModel(self, param):
        fileName = self.__dirModel + self.__fileModel
        classifier = joblib.load(fileName)

        print("Com o modelo salvo: ")
        print("Classificar ", param)
        print(classifier.predict(param))

        print("Classificar com proba", param)
        print(classifier.predict_proba(param))


    def testModel(self, fileIn, separator, blocks):

        path = self.__dirData
        df = pd.read_csv(path + fileIn, sep=separator)

        # Divide os dados em dois conjuntos: Atributos e Classes
        attributes = df.drop('Fusao_de_Vertebras', axis=1)
        classes = df['Fusao_de_Vertebras']

        X_train, X_test, y_train, y_test = train_test_split(attributes, classes, test_size=0.20)  

        classifier = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    
    def validateModel(self, classifier, X_test, y_test, y_pred, type, blocks):

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
        


        '''
        teste z
            * média (mean) e desvio padrão (std)
        testes de estatística kappa 
            * maior, melhor

        '''
