# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 00:18:32 2024

@author: terzi
"""
# ev fiyati tahmin 

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv('linear_regression_data2.csv')

sale_price = data[['SalePrice']]

neighborhood = data[['Neighborhood']]

data = data.iloc[:,0:4]

neighborhood = OneHotEncoder().fit_transform(neighborhood).toarray()
neighborhoodDF = pd.DataFrame(data=neighborhood,index=range(10),columns=['BrkSide','CollgCr','Crawfor','Mitchel','NWAmes','NoRidge','OldTown','Somerst','Veenker'])

data = pd.concat([data,neighborhoodDF],axis=1)

x_train, x_test, y_train, y_test = train_test_split(data,sale_price,test_size=0.33,random_state=0)

lr = LinearRegression().fit(x_train, y_train)

predict = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()


plt.scatter(x_train['GrLivArea'], y_train, color='blue', label='Eğitim Verisi') # "GrLivArea" özelliğinin fiyatlar üzerindeki etkisi
plt.plot(x_test['GrLivArea'], predict, color='red', label='test verisi') # GrLivArea özelliğinin fiyata etkisinin tahmini

plt.title('GrLivArea özelliğinin fiyata etkisi')
plt.xlabel('GrLivArea')
plt.ylabel('Sale Price')

plt.legend()
plt.show()