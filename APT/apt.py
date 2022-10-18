import numpy as np
import pandas as pd
from datetime import datetime

import pandas_datareader.data as pdr

import yfinance as yf

import matplotlib.pyplot as plt
plt.style.use('seaborn')

#Import Data

start = datetime(2021, 4, 1)
end = datetime(2021, 8, 31)

stock = yf.Ticker('AAPL').history(start=start, end=end) #Apple Inc.
market = yf.Ticker('SPY').history(start=start, end=end) #S&P 500 index.
vix = yf.Ticker('VXX').history(start=start, end=end)   #Volatility index. La volatilità attesta su 30 giorni basata sul prezzo delle opzioni the S&P 500
dxy = yf.Ticker('UUP').history(start=start, end=end) #Dollar index. indice del dollaro pesato su un basket di valute
junk = yf.Ticker('JNK').history(start=start, end=end)   #Junk bond index. Indice degli high yeld bond

#Preparazione dati di training e analisi dei dati con correlazione e diagrammi di dispersione
#Creat target dataframe
target = pd.DataFrame()
target['return'] = stock['Close'].pct_change(1)*100 #i dati sono ritardati di un giorno per evitare bias
target = target.dropna() #Drop NA nella prima riga
target.head()

#Create features dataframe
features = pd.DataFrame()
features['market'] = market['Close'].pct_change(1)*100
features['vix'] = vix['Close'].diff() #VIX è l'indice di volatilità e viene misurato in termini percentuali, quindi devi solo prendere la differenza
features['dxy'] = dxy['Close'].pct_change(1)*100
features['junk'] = junk['Close'].pct_change(1)*100
features = features.dropna() #Drop NA in the first row
features.head()

#Studio le correlazioni con un dataframe paralallelo

dataset = pd.DataFrame()
dataset['return'] = stock['Close'].pct_change(1)*100
dataset['market'] = market['Close'].pct_change(1)*100
dataset['vix'] = vix['Close'].diff() #VIX è l'indice di volatilità e viene misurato in termini percentuali, quindi devi solo prendere la differenza
dataset['dxy'] = dxy['Close'].pct_change(1)*100
dataset['junk'] = junk['Close'].pct_change(1)*100
dataset = dataset.dropna() #Drop NA in the first row
dataset.head()
print("Grado di correlazione")
print(dataset.corr())
print()

import seaborn as sns
sns.heatmap(dataset.corr(), xticklabels=dataset.columns, yticklabels=dataset.columns)
plt.show()
# diagrammi di dispersione
sns.pairplot(dataset)
plt.show()

#Train effettivo con regressione lineare multipla
#Eseguiamo una regressione lineare multipla
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

regression = LinearRegression()
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = 0.25, random_state=0)

#Faccio il training con il 75% dei dati
model = regression.fit(features_train, target_train)

print("Model Intercept:", model.intercept_)
print("Model Coefficients:", model.coef_)

#Evaluation
#Posso usare le matrice R2
y_pred = regression.predict(features_test)
error = mean_squared_error(target_test,y_pred)
print("Error:", error)
score = r2_score(target_test,y_pred)
print("Score:", score)

print("Training score: ", model.score(features_train, target_train))
print("Test score: ", model.score(features_test, target_test))

#LASSO L1 REGULARIZATION
#Rimuovo le feature meno importanti con L1 Lasso regularization
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler() #devo standardizzare gli input in modo che abbiano media a zero a varianza unitaria
features_standardized = scaler.fit_transform(features) #Ogni caratteristica ora ha media a zero e varianza unitaria
lasso_regression = Lasso (alpha = 0.2) #alpha (is the new lambda) è l'iperparametro (default = 1). Aumentare alpha aumenta la regolarizzazione
features_standardized_train, features_standardized_test, target_train, target_test = train_test_split(features_standardized, target, test_size=0.25, random_state=0)
lasso_model = lasso_regression.fit(features_standardized_train, target_train)

print("Lasso model intercept:", lasso_model.intercept_)
print("Lasso model coefficients:", lasso_model.coef_)

print("Lasso training score: ", lasso_model.score(features_standardized_train, target_train))
print("Lasso test score: ", lasso_model.score(features_standardized_test, target_test))

#RIDGE L2 REGULARIZATION
#Riduco gli effetti di tutti le features
from sklearn.linear_model import Ridge

ridge_regression = Ridge (alpha=10) #alpha is a hyperparameter. Increasing it increases regularization

features_standardized_train, features_standardized_test, target_train, target_test = train_test_split(features_standardized, target, test_size=0.25, random_state=0)
ridge_model = ridge_regression.fit(features_standardized_train, target_train)

print("Ridge model intercept:", ridge_model.intercept_)
print("Ridge model coefficients:", ridge_model.coef_)

print("Ridge training score: ", ridge_model.score(features_standardized_train, target_train))
print("Ridge test score: ", ridge_model.score(features_standardized_test, target_test))

#BONUS ELASTICNET
from sklearn.linear_model import  ElasticNet
# uso il modello ElasticNet che utilizza entrambe le regolarizzazioni
# NOTA: il parametro l1_ratio indica a quale regolarizzazione dare più importanza
# 0.5 -> sia L1 che L2 sono utilizzate con lo stesso peso nella regolarizzazione complessiva
# >0.5 -> L1 ha un peso maggiore nella regolarizzazione complessiva
# <0.5 -> L1 ha un peso minore nella regolarizzazione complessiva

enet_regression = ElasticNet(alpha=0.1, l1_ratio=0.5)

features_standardized_train, features_standardized_test, target_train, target_test = train_test_split(features_standardized, target, test_size=0.25, random_state=0)
enet_regression = enet_regression.fit(features_standardized_train, target_train)

print("Ridge model intercept:", enet_regression.intercept_)
print("Ridge model coefficients:", enet_regression.coef_)

print("Ridge training score: ", enet_regression.score(features_standardized_train, target_train))
print("Ridge test score: ", enet_regression.score(features_standardized_test, target_test))