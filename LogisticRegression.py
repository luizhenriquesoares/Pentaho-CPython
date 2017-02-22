# python script

from pandas import read_csv
from pandas import DataFrame
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import csv

array = df.values

# Separando o array em componentes de input e output
X = array[:,0:8]
Y = array[:,8]

# Definindo o tamanho do conjunto de dados
teste_size = 0.33
seed = 7

# Dividindo os dados em treino e teste
X_treino, X_teste, Y_treino, Y_teste = cross_validation.train_test_split(X, Y,
                                                                     test_size = teste_size,

# Criando o modelo
model = LogisticRegression()
model.fit(X_treino, Y_treino)

# Score
acuracia = model.score(X_teste, Y_teste)

acuracia = acuracia * 100.0
print(acuracia)

