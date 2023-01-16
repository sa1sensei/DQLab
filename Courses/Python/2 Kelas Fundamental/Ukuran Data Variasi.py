# # # # # # # # # Membaca Dataset

# In[1]:


# Import numpy sebagai aliasnya np
import numpy as np

# Baca dataset 
tinggi_badan = np.loadtxt("https://storage.googleapis.com/dqlab-dataset/survei_tinggi_badan.txt", skiprows=1)

# Cetak ukuran dataset dan isinya
print("Ukuran data tinggi_badan:", tinggi_badan.shape)
print("Data tinggi_badan (cm):\n", tinggi_badan)



# # # # # # # # # Rentang Data (Range)

# In[2]:

# Import numpy sebagai aliasnya np
import numpy as np
# Baca dataset 
tinggi_badan = np.loadtxt("https://storage.googleapis.com/dqlab-dataset/survei_tinggi_badan.txt", skiprows=1)

# Tentukan rentang data
print("Range data:")
print("  min:", tinggi_badan.min())
print("  max:", tinggi_badan.max())




# # # # # # # # # Visualisasi Data dengan Swarmplot

# In[3]:


# Import numpy
import numpy as np
# Baca dataset
tinggi_badan = np.loadtxt("https://storage.googleapis.com/dqlab-dataset/survei_tinggi_badan.txt", skiprows=1)

# Import matplotlib.pyplot dan seaborn sebagai aliasnya
import matplotlib.pyplot as plt
import seaborn as sns

# Visualisasikan dengan swarmplot variabel tinggi_badan dengan ukuran marker 8
fig, ax = plt.subplots(figsize=(10,4)) 
sns.swarmplot(x=tinggi_badan, size=8, color="darkcyan", edgecolor=None, ax=ax)
ax.grid(axis="x")
plt.xlabel("Tinggi badan (cm)", fontsize=14)
plt.tight_layout()
plt.show()



# # # # # # # # # Kuartil Data dengan numpy.percentile

# In[4]:

# Import numpy
import numpy as np
# Baca dataset 
tinggi_badan = np.loadtxt("https://storage.googleapis.com/dqlab-dataset/survei_tinggi_badan.txt", skiprows=1)

# Perhitungan Q1, Q2, dan Q3 satu persatu
print("Perhitungan Q1, Q2, dan Q3 satu persatu")
Q1 = np.percentile(tinggi_badan, 25)
Q2 = np.percentile(tinggi_badan, 50)
Q3 = np.percentile(tinggi_badan, 75)
print("Kuartil 1 (Q1):", Q1)
print("Kuartil 2 (Q2):", Q2)
print("Kuartil 3 (Q3):", Q3)

# Perhitungan Q1, Q2, dan Q3 sekaligus
print("\nPerhitungan Q1, Q2, dan Q3 sekaligus")
Q123 = np.percentile(tinggi_badan, [25, 50, 75])
print("[Q1, Q2, Q3]:", Q123)


# # # # # # # # # Kuartil Data dengan numpy.quantile

# In[5]:

# Import numpy
import numpy as np
# Baca dataset 
tinggi_badan = np.loadtxt("https://storage.googleapis.com/dqlab-dataset/survei_tinggi_badan.txt", skiprows=1)

# Perhitungan Q1, Q2, dan Q3 satu persatu
print("Perhitungan Q1, Q2, dan Q3 satu persatu")
Q1 = np.quantile(tinggi_badan, 0.25)
Q2 = np.quantile(tinggi_badan, 0.5)
Q3 = np.quantile(tinggi_badan, 0.75)
print("Kuartil 1 (Q1):", Q1)
print("Kuartil 2 (Q2):", Q2)
print("Kuartil 3 (Q3):", Q3)

# Perhitungan Q1, Q2, dan Q3 sekaligus
print("\nPerhitungan Q1, Q2, dan Q3 sekaligus")
Q123 = np.quantile(tinggi_badan, [0.25, 0.5, 0.75])
print("[Q1, Q2, Q3]:", Q123)



# # # # # # # # # Nilai Upper dan Lower Whisker

# In[6]:

# Import numpy
import numpy as np
# Baca dataset 
tinggi_badan = np.loadtxt("https://storage.googleapis.com/dqlab-dataset/survei_tinggi_badan.txt", skiprows=1)

# Perhitungan Q1, Q2, dan Q3
Q1, Q2, Q3 = np.percentile(tinggi_badan, [25, 50, 75])
# Jarak antar kuartil (inter quartile range, IQR)
IQR = Q3 - Q1
print("Jarak antar kuartil (IQR):", IQR)
print("\nlower whisker:", Q1 - 1.5*IQR)
print("           Q1:", Q1)
print("           Q2:", Q2)
print("           Q3:", Q3)
print("upper whisker:", Q3 + 1.5*IQR)


# # # # # # # # # Visualisasi Data dengan Box Plot

# In[7]:

# Import numpy
import numpy as np
# Baca dataset
tinggi_badan = np.loadtxt("https://storage.googleapis.com/dqlab-dataset/survei_tinggi_badan.txt", skiprows=1)

# Import matplotlib.pyplot dan seaborn sebagai aliasnya
import matplotlib.pyplot as plt
import seaborn as sns

ig, ax = plt.subplots(figsize=(10,4))
# Plotkanlah boxplot pada data tinggi_badan dengan nilai whis 1.5
sns.boxplot(x=tinggi_badan, whis=1.5, color="greenyellow", ax=ax)
# Plotkan kembali swarmplot
sns.swarmplot(x=tinggi_badan, size=8, color="darkcyan", edgecolor=None, ax=ax)
ax.grid(axis="x")
plt.xlabel("Tinggi badan (cm)", fontsize=14)
plt.tight_layout()
plt.show()


# # # # # # # # # Menentukan Persentil

# In[8]:

# Import numpy
import numpy as np
# Baca dataset 
tinggi_badan = np.loadtxt("https://storage.googleapis.com/dqlab-dataset/survei_tinggi_badan.txt", skiprows=1)

# Persentil ke-5 dan ke-95
print("Menggunakan np.percentile")
P5, P95 = np.percentile(tinggi_badan, [5, 95])
print("  Persentil ke-5  (P5) :", P5)
print("  Persentil ke-95 (P95):", P95)

print("\nMenggunakan np.quantile")
P5, P95 = np.quantile(tinggi_badan, [0.05, 0.95])
print("  Persentil ke-5  (P5) :", P5)
print("  Persentil ke-95 (P95):", P95)

# # # # # # # # # Varians dan Standar Deviasi dengan Python

# In[9]:

# Import numpy
import numpy as np
# Baca dataset 
tinggi_badan = np.loadtxt("https://storage.googleapis.com/dqlab-dataset/survei_tinggi_badan.txt", skiprows=1)

def py_rata_rata(data):
    return sum(data) / len(data)

def py_varians(data, k=1):
    rerata, var = py_rata_rata(data), 0
    for d in data:
        var += (d - rerata) ** 2
    return var / (len(data) - k)

def py_standar_deviasi(data, k=1):
    return py_varians(data, k=k) ** 0.5

print("Menggunakan user-defined function pada array tinggi_badan")
print("  unbiased varians        :", py_varians(tinggi_badan))
print("  biased varians          :", py_varians(tinggi_badan, k=0))
print("  unbiased standar deviasi:", py_standar_deviasi(tinggi_badan))
print("  biased standar deviasi  :", py_standar_deviasi(tinggi_badan, k=0))

print("\nMenggunakan method .var() dan .std() pada array tinggi_badan")
print("  unbiased varians        :", tinggi_badan.var(ddof=1))
print("  biased varians          :", tinggi_badan.var())
print("  unbiased standar deviasi:", tinggi_badan.std(ddof=1))
print("  biased standar deviasi  :", tinggi_badan.std())


# # # # # # # # # Varians dan Standar Deviasi dengan Numpy

# In[10]:

# Import numpy
import numpy as np
# Baca dataset 
tinggi_badan = np.loadtxt("https://storage.googleapis.com/dqlab-dataset/survei_tinggi_badan.txt", skiprows=1)

def np_rata_rata(data):
    return data.sum() / len(data)

def np_varians(data, k=1):
    rerata = np_rata_rata(data)
    return ((data - rerata) ** 2).sum() / (len(data) - k)

def np_standar_deviasi(data, k=1):
    return np_varians(data, k=k) ** 0.5

print("Menggunakan user-defined function pada array tinggi_badan")
print("  unbiased varians        :", np_varians(tinggi_badan))
print("  biased varians          :", np_varians(tinggi_badan, k=0))
print("  unbiased standar deviasi:", np_standar_deviasi(tinggi_badan))
print("  biased standar deviasi  :", np_standar_deviasi(tinggi_badan, k=0))

print("\nMenggunakan method .var() dan .std() pada array tinggi_badan")
print("  unbiased varians        :", tinggi_badan.var(ddof=1))
print("  biased varians          :", tinggi_badan.var())
print("  unbiased standar deviasi:", tinggi_badan.std(ddof=1))
print("  biased standar deviasi  :", tinggi_badan.std())


# # # # # # # # # Problem #1

# In[11]:

# Import pandas sebagai aliasnya yaitu pd
import pandas as pd

# Baca dataset
tinggi_badan = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/survei_tinggi_badan.txt")
# Cetak type(tinggi_badan) dan data frame tinggi_badan sendiri
print("type(tinggi_badan):", type(tinggi_badan))
print(tinggi_badan)

# Hitung statistik deskriptif tinggi_badan
statistik_deskriptif = tinggi_badan.describe()
# Cetak statistik deskriptif tinggi_badan
print("\nStatistik deskriptif:\n", statistik_deskriptif)

# Tentukan IQR
Q1 = statistik_deskriptif["tinggi badan (cm)"]["25%"]
Q3 = statistik_deskriptif["tinggi badan (cm)"]["75%"]
IQR = Q3 - Q1
#Cetak IQR
print("\nIQR:", IQR)

# Persentil ke-5 dan ke-95
percentile = tinggi_badan.quantile(q=[0.05, 0.95])
print("\nPersentil ke-5 dan ke-95:\n", percentile)


# # # # # # # # # Problem #2

# In[12]:

# Import pandas sebagai aliasnya yaitu
import pandas as pd

# Baca dataset
tinggi_badan = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/survei_tinggi_badan.txt")
print("Statistik deskriptif data awal:\n", tinggi_badan.describe().T)

# Modifikasi data maksimum tinggi badan menjadi 186 cm
tinggi_badan_mod = tinggi_badan.replace(tinggi_badan.max(), 186)
print("\nStatistik deskriptif data modifikasi:\n", tinggi_badan_mod.describe().T)

# Import matplotlib.pyplot dan seaborn 
import matplotlib.pyplot as plt
import seaborn as sns
fig, axs = plt.subplots(2, 1, figsize=(12,8), sharex=True)
# Plot data awal
sns.boxplot(data=tinggi_badan, x="tinggi badan (cm)", whis=1.5, color="yellowgreen", ax=axs[0])
sns.swarmplot(data=tinggi_badan, x="tinggi badan (cm)", size=8, color="darkcyan", edgecolor=None, ax=axs[0])
axs[0].grid(axis="x")
axs[0].set_xlabel(None)
axs[0].set_title("Data awal", color="darkcyan", fontsize=16)

# Plot data modifikasi
sns.boxplot(data=tinggi_badan_mod, x="tinggi badan (cm)", whis=1.5, color="yellowgreen", ax=axs[1])
sns.swarmplot(data=tinggi_badan_mod, x="tinggi badan (cm)", size=8, color="darkcyan", edgecolor=None, ax=axs[1])
axs[1].grid(axis="x")
axs[1].set_xlabel(None)
axs[1].set_title("Data modifikasi", color="darkcyan", fontsize=16)

plt.xlabel("Tinggi badan (cm)", fontsize=14)
plt.tight_layout()
plt.show()