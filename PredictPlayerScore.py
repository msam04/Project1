
# coding: utf-8

# In[60]:


import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from math import sqrt

# Create your connection.
cnx = sqlite3.connect('C://Users//Monica//Python_Scripts//PredictPlayer//database.sqlite')
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

x_train = df[input_column_list]
y_train = df[output_column_list]

scaler = StandardScaler().fit(x_train)
x_t = scaler.transform(x_train)  

pca = PCA(n_components = 30)
fit = pca.fit(x_t)
x_t2 = pca.fit_transform(x_t)


lnr = LinearRegression()
model = lnr.fit(x_t2,y_train)
predictions = lnr.predict(x_t2)
lnr.score(x_t2,y_train)
#print(cross_val_score(lnr, X_new, y,cv = 5).mean()) 

dtr = DecisionTreeRegressor()
model = dtr.fit(x_t2, y_train)
dtr.score(x_t2, y_train)

