# # # # # # # # # Load Library

# In[1]:

# memuat numpy sebagai np
import numpy as np
 
# memuat pandas sebagai pd
import pandas as pd

# # # # # # # # # Load Dataset

# In[2]:

import pandas as pd
# memuat data bernama 'dataset_statistics.csv' dan memasukkan hasilnya ke dalam 'raw_data'
raw_data = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/dataset_statistic.csv", sep=';')

# # # # # # # # # Inspeksi Data

# In[3]:

import pandas as pd
raw_data = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/dataset_statistic.csv", sep=';')

print (raw_data)
# melihat 10 data pada baris pertama
print (raw_data.head(10))

# melihat 5 data pada baris terakhir
print (raw_data.tail())



# # # # # # # # # Metode Shape

# In[4]:

import pandas as pd
raw_data = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/dataset_statistic.csv", sep=';')

# melihat dimensi dari raw_data
print (raw_data.shape)

# mengambil jumlah data
print (raw_data.shape[0])

# # # # # # # # # Melihat Kolom Dalam Dataset

# In[5]:

import pandas as pd
raw_data = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/dataset_statistic.csv", sep=';')

print(raw_data.columns)

# # # # # # # # # Metode Isna

# In[6]:

import pandas as pd
raw_data = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/dataset_statistic.csv", sep=';')

print (raw_data.isna())
print (raw_data.isna().sum())

# # # # # # # # # Metode Describe

# In[7]:

import pandas as pd
raw_data = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/dataset_statistic.csv", sep=';')

print (raw_data.describe())

# Mencari nilai maksimum dari tiap kolom
raw_data.max()
 
# Mencari nilai maksimum dari kolom 'Harga'
raw_data['Harga'].max()
 
# Mencari nilai minimum dari kolom 'Harga'
raw_data['Harga'].min()

# # # # # # # # # Metode Sum

# In[8]:

import pandas as pd
raw_data = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/dataset_statistic.csv", sep=';')

# menghitung jumlah dari semua kolom
print (raw_data.sum())
 
# menghitung jumlah dari semua kolom bertipe data numerik saja
raw_data.sum(numeric_only=True)
 
# menghitung jumlah dari kolom 'Harga' dan 'Pendapatan'
raw_data[['Harga', 'Pendapatan']].sum()

# # # # # # # # # Manipulasi Dataframe - Memilih Kolom dan Baris

# In[9]:

import pandas as pd
raw_data = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/dataset_statistic.csv", sep=';')

# Memilih kolom 'Pendapatan' saja
print (raw_data['Pendapatan'])
 
# Memilih kolom 'Jenis Kelamin' dan 'Pendapatan'
print (raw_data[['Jenis Kelamin', 'Pendapatan']])


# # # # # # # # # Metode Loc

# In[10]:

import pandas as pd
raw_data = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/dataset_statistic.csv", sep=';')

# mengambil data dari baris ke-0 sampai baris ke-(10-1) atau baris ke-9
print(raw_data[:10])
 
# mengambil data dari baris ke-3 sampai baris ke-(5-1) atau baris ke-4
print(raw_data[3:5])
 
# mengambil data pada baris ke-1, ke-3 dan ke-10
print(raw_data.loc[[1,3,10]])

# Mengambil kolom 'Jenis Kelamin' dan 'Pendapatan' dan ambil baris ke-1 sampai ke-9
print(raw_data[['Jenis Kelamin', 'Pendapatan']][1:10])
 
# Mengambil kolom 'Harga' dan 'Tingkat Kepuasan' dan ambil baris ke-1, ke-10 dan ke-15
print(raw_data[['Harga', 'Tingkat Kepuasan']].loc[[1,10,15]])

# # # # # # # # # Rata-rata (Mean)

# In[11]:

import numpy as np
import pandas as pd
raw_data = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/dataset_statistic.csv", sep=';')

# mengambil hanya data untuk produk 'A'
produk_A = raw_data[raw_data['Produk'] == 'A']
 
# menghitung rerata pendapatan menggunakan method .mean pada objek pandas DataFrame
print (produk_A['Pendapatan'].mean())
 
# menghitung rerata pendapatan menggunakan method .mean pada objek pandas DataFrame dengan numpy
print (np.mean(produk_A['Pendapatan']))


# # # # # # # # # Median
# In[12]:

import numpy as np
import pandas as pd
raw_data = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/dataset_statistic.csv", sep=';')
produk_A = raw_data[raw_data['Produk'] == 'A']

print (raw_data)
# Hitung median dari pendapatan menggunakan pandas
print (produk_A['Pendapatan'].median())

# Hitung median dari pendapatan menggunakan numpy
print (np.median(produk_A['Pendapatan']))

# # # # # # # # # Modus
# In[13]:

import pandas as pd
raw_data = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/dataset_statistic.csv", sep=';')

# Melihat jumlah dari masing-masing produk
print (raw_data['Produk'].value_counts())

# # # # # # # # # Kuantil
# In[14]:

import numpy as np
import pandas as pd
raw_data = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/dataset_statistic.csv", sep=';')

# mencari median atau 50% dari data menggunakan pandas
print (raw_data['Pendapatan'].quantile(q = 0.5))

# mencari median atau 50% dari data menggunakan pandas
print (np.quantile(raw_data['Pendapatan'], q=0.5))

# # # # # # # # # Agregasi Data dengan method .agg()

# In[15]:

import numpy as np
import pandas as pd
raw_data = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/dataset_statistic.csv", sep=';')

# menghitung rerata dan median usia (age) dan insulin (insu)
print (raw_data[['Pendapatan', 'Harga']].agg([np.mean, np.median]))

# menghitung rerata dan median Pendapatan dan Harga dari tiap produk
print (raw_data[['Pendapatan', 'Harga', 'Produk']].groupby('Produk').agg([np.mean, np.median]))

# # # # # # # # # Proporsi Kategori

# In[16]:

import pandas as pd
raw_data = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/dataset_statistic.csv", sep=';')

# cari proporsi tiap Produk
print (raw_data['Produk'].value_counts()/raw_data.shape[0])

# # # # # # # # # Ukuran Sebaran pada Data Interval dan Rasio

# In[17]:

import pandas as pd
raw_data = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/dataset_statistic.csv", sep=';')

# Cari nilai rentang dari kolom 'Pendapatan'
print (raw_data['Pendapatan'].max() - raw_data['Pendapatan'].min())


# # # # # # # # # Variansi
# In[18]:

import numpy as np
import pandas as pd
raw_data = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/dataset_statistic.csv", sep=';')

# menghitung variansi umur menggunakan method .var() dari pandas
print (raw_data['Pendapatan'].var())
 
# menghitung variansi umur menggunakan method .var() dari numpy
print (np.var(raw_data['Pendapatan']))

# mengatur variansi populasi dengan method `.var()` dari pandas
print (raw_data['Pendapatan'].var(ddof=0))



# # # # # # # # # Deviasi Baku (Standard Deviation)

# In[19]:

import numpy as np
import pandas as pd
raw_data = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/dataset_statistic.csv", sep=';')

# menghitung deviasi baku sampel pendapatan menggunakan method std() dari pandas
print (raw_data['Pendapatan'].std())
 
# menghitung deviasi baku sampel pendapatan menggunakan method std() dari numpy
print (np.std(raw_data['Pendapatan'], ddof = 1))

# # # # # # # # # Menghitung Korelasi

# In[20]:

import pandas as pd
raw_data = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/dataset_statistic.csv", sep=';')

# menghitung korelasi dari setiap pasang variabel pada raw_data
print (raw_data.corr())

# mencari korelasi 'kendall' untuk tiap pasang variabel
print (raw_data.corr(method='kendall'))
 
# mencari korelasi 'spearman' untuk tiap pasang variabel
print (raw_data.corr(method='spearman'))

