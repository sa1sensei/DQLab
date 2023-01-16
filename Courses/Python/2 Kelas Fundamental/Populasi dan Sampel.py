# # # # # # # # # Simple Random Sampling Menggunakan Python

# In[1]:

# Import modul random
import random 

# Set seed sebagai bilangan bulat 0, dan dapat diganti
# dengan bilangan bulat lainnya, sesuai dengan instruksi Senja
random.seed(1234)

# Ambil sampel dalam rentang butir data, yaitu 1 s/d 120
# Inisialisasi sampel
sampel = []
# Looping sebanyak sampel yang ingin ditarik yaitu 10% (12 butir data)
for i in range(12): 
    sampel.append(random.randint(1, 120))
# Cetaklah sampel
print("sampel:", sampel)



# # # # # # # # # Simple Random Sampling Menggunakan Numpy

# In[2]:


# Import numpy sebagai aliasnya np
import numpy as np

# Set seed sebagai bilangan bulat 0, dan dapat diganti
# dengan bilangan bulat lainnya
np.random.seed(0)

# Ambil sampel dalam rentang butir data, yaitu 1 s/d 120,
# sebanyak 10% (12 butir data)
sampel = np.random.randint(1, 121, size=12)
# Cetaklah sampel
print("sampel:", sampel)


# # # # # # # # # Menentukan Jumlah Sampel

# In[3]:

# Import math
import math
# Jumlah anggota populasi
N = 8963
# Proporsi
p = 0.25
# z-score
z = 1.96
# Margin of error
e = 0.05
# Perhitungan ukuran sampel, n
n_aksen = z**2 * p * (1 - p) / e**2
n = n_aksen / (1 + (n_aksen / N))
# Cetak jumlah sampel
print("Jumlah sampel:", math.ceil(n))



# # # # # # # # # Problem 1

# In[4]:

# Import numpy sebagai aliasnya np
import numpy as np
# Jumlah anggota populasi
N = 8963 
# Proporsi
p = 0.25
# Selang kepercayaan (confidence interval)
ci = [0.70, 0.75, 0.80, 0.85, 0.92, 0.95, 0.96, 0.98, 0.99, 0.999]
# z-score
z = [1.04,1.15,1.28,1.44,1.75,1.96,2.05,2.33,2.58,3.29]
z = np.array(z)
# Margin of error
e = 0.05
# Perhitungan ukuran sampel, n
n_aksen = z**2 * p * (1 - p) / e**2
n = np.ceil(n_aksen / (1 + (n_aksen / N)))
# Cetak ukuran sampel untuk setiap variasi selang kepercayaan
print("Ukuran sampel untuk setiap variasi selang kepercayaan")
print("+--------------------+---------------+")
print("| Selang kepercayaan | Jumlah sampel |")
print("+--------------------+---------------+")
for ci_, n_ in zip(ci, n):
    print("| %17.3f  | %13d |" % (ci_, n_))
print("+--------------------+---------------+")


# # # # # # # # # Problem 2

# In[5]:

# Import numpy sebagai aliasnya
import numpy as np
# Jumlah anggota populasi
N = 8963
# Proporsi
p = 0.25
# Selang kepercayaan (confidence interval) ci = 0.95
# dengan z-score sebesar
z =  1.96
# Margin of error
e = np.array([ 0.01, 0.02, 0.05, 0.10, 0.20, 0.25, 0.33, 0.50])
# Perhitungan ukuran sampel, n
n_aksen = z**2 * p * (1 - p) / e**2
n = np.ceil(n_aksen / (1 + (n_aksen / N)))
# Cetak ukuran sampel untuk setiap variasi margin galat
print("Ukuran sampel untuk setiap variasi margin galat")
print("+--------------+---------------+")
print("| Margin galat | Jumlah sampel |")
print("+--------------+---------------+")
for e_, n_ in zip(e, n):
    print("| %12.2f | %13d |" % (e_, n_))
print("+--------------+---------------+")

