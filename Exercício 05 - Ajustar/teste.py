import Diabetes_DecisionTree as model


#df = pd.read_csv('Dados\\diabetes.csv', sep=',')


exec = model.Executar()

exec.setModel('diabetes.csv', 'diabetes_Normal.csv', ',')
exec.setModel('Diabetes.csv', ';', 30)
