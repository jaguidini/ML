import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn import svm

diabetes = pd.read_csv("dados/diabetes.csv", delimiter=',')

attributes = diabetes.drop('class', axis=1)
classes = diabetes['class']


x_train, x_test, y_train, y_test = train_test_split(attributes, classes, test_size=0.20, random_state=0)

#(Vector Machine)
classifier = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)

print("cros_val_score")
scores = cross_val_score(classifier, x_test, y_test, cv=10)
print(scores)
print("Precisao media:", scores.mean())

print("cros_validate")
scores = cross_validate(classifier, x_test, y_test, cv=10)
print(scores)
print("Precisao media:", scores['test_score'].mean())
