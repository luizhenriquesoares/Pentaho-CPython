# Avaliação usando Cross Validation

# Import dos módulos
from pandas import read_csv
from pandas import DataFrame
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
import csv

# Carregando os dados
url = "pima-data.csv"
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
resultado = cross_validation.cross_val_score(modelo, standardX, Y, cv = kfold)

score = resultado.mean()*100.0
predict  = cross_validation.cross_val_predict(modelo, standardX, Y, cv = kfold)
matrix = confusion_matrix(Y, predict)

def writeCsvVarObservadoras(df, name):
    df = DataFrame(df, columns = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'])
    df.index.name = 'id'
    df.to_csv(name, sep=',', encoding='utf-8')

def writeCsvVarPredict(df, name):
    df = DataFrame(df, columns = ['predict-class'])
    df.index.name = 'id'
    df.to_csv(name, sep=',', encoding='utf-8')

writeCsvVarPredict(predict, 'predict.csv') 
writeCsvVarObservadoras(array,'observadoras.csv') 



#teste


