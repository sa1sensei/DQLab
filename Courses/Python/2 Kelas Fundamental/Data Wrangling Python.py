# # # # # # # # # Menampilkan informasi statistik dengan Numpy
# In[1]:

import pandas as pd

csv_data = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/shopping_data.csv")

print(csv_data.describe(exclude=["O"]))



# # # # # # # # # Melakukan pengecekan untuk nilai NULL yang ada
# In[2]:

import pandas as pd

csv_data = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/shopping_data_missingvalue.csv")

print(csv_data.isnull().values.any())



# # # # # # # # # Mengisi dengan Mean
# In[3]:


import pandas as pd

csv_data = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/shopping_data_missingvalue.csv")
print(csv_data.mean())
print("Dataset yang masih terdapat nilai kosong ! :")
print(csv_data.head(10))

csv_data=csv_data.fillna(csv_data.mean())
print("Dataset yang sudah diproses Handling Missing Values dengan Mean :")
print(csv_data.head(10))


# # # # # # # # # Mengisi dengan Median
# In[4]:

import pandas as pd

csv_data = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/shopping_data_missingvalue.csv")
print("Dataset yang masih terdapat nilai kosong ! :")
print(csv_data.head(10))

csv_data=csv_data.fillna(csv_data.median())
print("Dataset yang sudah diproses Handling Missing Values dengan Median :")
print(csv_data.head(10))


# # # # # # # # # Praktek Normalisasi menggunakan Scikit Learn pada Python
# In[5]:


import pandas as pd
import numpy as np
from sklearn import preprocessing

csv_data = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/shopping_data.csv")
array = csv_data.values
X = array[:,2:5] #memisahkan fitur dari dataset. 
Y = array[:,0:1]  #memisahkan class dari dataset

dataset=pd.DataFrame({"Customer ID":array[:,0],"Gender":array[:,1],"Age":array[:,2],"Income":array[:,3],"Spending Score":array[:,4]})
print("dataset sebelum dinormalisasi :")
print(dataset.head(10))

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1)) #inisialisasi normalisasi MinMax
data = min_max_scaler.fit_transform(X) #transformasi MinMax untuk fitur
dataset = pd.DataFrame({"Age":data[:,0],"Income":data[:,1],"Spending Score":data[:,2],"Customer ID":array[:,0],"Gender":array[:,1]})

print("dataset setelah dinormalisasi :")
print(dataset.head(10))
