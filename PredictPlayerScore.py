
# coding: utf-8

# In[ ]:


import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from math import sqrt

from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import zipfile


zip_ref = zipfile.ZipFile('soccer.zip', 'r')
zip_ref.extractall()
zip_ref.close()

#print(isfile('database.sqlite'))


# Create your connection.
cnx = sqlite3.connect('database.sqlite')
df = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)
#print(df.head())
print(df.shape)
print(df.columns)
#Removing columns that do not contribute to score
df.drop(columns = ['id', 'player_fifa_api_id', 'player_api_id', 'date'], inplace = True)
print(df.shape)

#Replace NaNs with 0s
df.fillna(0, inplace = True)
#Remove duplicates
df.drop_duplicates(inplace = True)

#Print out non-numeric type columns
column_list = df.columns.values.tolist()
for i in range(len(column_list)):
    if not(np.issubdtype(df[column_list[i]].dtype, np.number)):
        print(column_list[i],"\n===========================")
        print (df[column_list[i]].unique())
    
#Replace nonsensical values with median or default values
df['preferred_foot'].replace(0, 'left', inplace = True)
df['attacking_work_rate'].replace([0,'None', 'le', 'norm', 'stoc', 'y'],'medium', inplace = True)
df['defensive_work_rate'].replace(['_0', 0, '5', 'ean', 'o', '1', 'ormal', '7', '2', '8', '4', 'tocky', '0', '3', '6', '9', 'es'], 'medium', inplace = True)

print("preferred_foot\n===============================")
print(df['preferred_foot'].unique())
print("attacking_work_rate\n===========================")
print(df['attacking_work_rate'].unique())
print("defensive_work_rate\n===============================")
print(df['defensive_work_rate'].unique())

#Change categorical values to numeric values
df['preferred_foot'] = df['preferred_foot'].map({'left': 0, 'right': 1})
df['defensive_work_rate'] = df['defensive_work_rate'].map({'low': 0, 'medium': 1, 'high' : 2})
df['attacking_work_rate'] = df['attacking_work_rate'].map({'low': 0, 'medium': 1, 'high' : 2})

#//divide with integer, discard remainder
#fig, axes = plt.subplots(len(df.columns)//3, 3, figsize=(8, 32))

#i = 0
#for triaxis in axes:
#    for axis in triaxis:
#        df.hist(column = df.columns[i], bins = 100, ax=axis)
#        i = i+1
        
#plt.show()

#fig, axes = plt.subplots(len(df.columns)//3, 3, figsize=(8, 32))

#i = 0
#for triaxis in axes:
#    for axis in triaxis:
#        df.boxplot(column = df.columns[i], ax=axis)
#        i = i+1
        
#plt.show()

#Performing PCA after scaling, because most of the variables have a somewhat normal distribution


input_column_list = column_list.copy()
input_column_list.remove('overall_rating')
output_column_list ='overall_rating'

x = df[input_column_list]
y = df[output_column_list]

scaler = StandardScaler().fit(x)
x1 = scaler.transform(x)  

pca = PCA(n_components = 30)
fit = pca.fit(x1)
x2 = pca.fit_transform(x1)

ts = int(np.floor(0.20 * df.shape[0]))


train_x, test_x, train_y, test_y = train_test_split(x2, y, 
                                                    test_size= ts,
                                                    random_state=0)

#lnr = LinearRegression()
#model = lnr.fit(x_t2,y_train)
#predictions = lnr.predict(x_t2)
#lnr.score(x_t2,y_train)
#print(cross_val_score(lnr, X_new, y,cv = 5).mean()) 

dtr = DecisionTreeRegressor()
model = dtr.fit(train_x, train_y)
pred_y = dtr.predict(test_x)
print(mean_squared_error(test_y, pred_y))
print(dtr.score(test_x, test_y))
print(cross_val_score(dtr, x2, y,cv = 5).mean())


