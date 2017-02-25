# Avaliação usando Cross Validation

# Import dos módulos
from pandas import read_csv
from pandas import DataFrame
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
import csv

# Carregando os dados
url = "http://datascienceacademy.com.br/blog/aluno/Python-Spark/Datasets/pima-data.csv"
colunas = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv(url, names = colunas)
array = df.values

# Separando o array em componentes de input e output
X = array[:,0:8]
Y = array[:,8]

# Gerando o novo padrão, aplicando pré-processamento
scaler = StandardScaler().fit(X)
standardX = scaler.transform(X)

# Definindo os valores para os folds
num_folds = 40
num_instances = len(X)
seed = 7

# Separando os dados em folds
kfold = cross_validation.KFold(n = num_instances, n_folds = num_folds, random_state = seed)
    
# Criando o modelo
modelo = LogisticRegression()
scores = cross_validation.cross_val_score(modelo, standardX, Y, cv = kfold)
predict  = cross_validation.cross_val_predict(modelo, standardX, Y, cv = kfold)

matrix = confusion_matrix(Y, predict)
print(matrix)
print("Acurácia Treino: %.3f%% (%.3f%%)" % (scores.mean()*100.0, scores.std() * 100.0))
print(accuracy_score(Y, predict) *100.0)