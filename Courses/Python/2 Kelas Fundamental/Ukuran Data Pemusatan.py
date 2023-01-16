# # # # # # # # # Membaca Data Survei

# In[1]:


# Import library pandas sebagai aliasnya pd
Import pandas as pd

# Membaca file survei_tinggi_badan.txt
df = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/survei_tinggi_badan.txt")

# Mencetak 5 data teratas
print(df.head(5))


# # # # # # # # # Mengubah Nama Kolom

# In[2]:


# import library pandas
import pandas as pd
# membaca file survei_tinggi_badan.txt
df = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/survei_tinggi_badan.txt")

# melakukan rename terhadap kolom "tinggi badan (cm)" menjadi "tinggi"
df.rename("tinggi badan (cm)" : "tinggi", axis=1, inplace=True)
print(df.head(5))


# # # # # # # # # Nilai Mean

# In[3]:

# import library pandas
import pandas as pd
# membaca file survei_tinggi_badan.txt
df = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/survei_tinggi_badan.txt")
# melakukan rename terhadap kolom "tinggi badan (cm)" menjadi "tinggi"
df.rename({'tinggi badan (cm)' : 'tinggi'}, axis=1, inplace=True)

# Mengecek apakah ada data yang bernilai null
print("Mengecek apakah ada data yang bernilai null")
print(df.isnull().any().any())

# Mengecek jumlah data
print("\nMengecek jumlah data")
print(df.shape)

# Mendapatkan nilai mean menggunakan fungsi mean yang disediakan pandas
print("\nMean:", df["tinggi"].mean())




# # # # # # # # # Nilai Median dan Modus

# In[4]:

# Import library pandas
import pandas as pd
# Membaca file survei_tinggi_badan.txt
df = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/survei_tinggi_badan.txt")
# melakukan rename terhadap kolom "tinggi badan (cm)" menjadi "tinggi"
df.rename({'tinggi badan (cm)' : 'tinggi'}, axis=1, inplace=True)

# Mendapatkan nilai median menggunakan fungsi median yang disediakan pandas
print("Median data")
print(">> Median: ".df["tinggi"].median())

# Mendapatkan nilai modus menggunakan fungsi mode yang disediakan pandas
print("\nModus data")
print(">> Modus: " ,df["tinggi"].mode())


# # # # # # # # # Membaca Dataset

# In[5]:

# import library pandas
import pandas as pd

#load dataset ke dataframe anscombe
anscombe = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/anscombe.csv")

# menampilkan seluruh baris data yang ada
print("Anscombe's Quartet")
print(anscombe)

# menampilkan mean dataset anscombe
print("\nMean dari Anscombe's Quartet")
print(anscombe.mean())

# # # # # # # # # Visualisasi Scatter

# In[6]:

# import library pandas
import pandas as pd
#load dataset ke dataframe anscombe
anscombe = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/anscombe.csv")

# import matplotlib untuk plotting
import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8,8))
# scatter plot (x1, y1), (x2, y2), (x3, y3), dan (x4, y4)
ax[0][0].scatter(anscombe.x1, anscombe.y1)
ax[0][1].scatter(anscombe.x2, anscombe.y2)
ax[1][0].scatter(anscombe.x3, anscombe.y3)
ax[1][1].scatter(anscombe.x4, anscombe.y4)

k = 1
for i in range(2):
	for j in range(2):
		ax[i][j].set_xlabel("x" + str(k), fontsize=12)
		ax[i][j].set_ylabel("y" + str(k), fontsize=12)
		ax[i][j].grid()
		k += 1
		
plt.tight_layout()
plt.show()

# # # # # # # # # Latihan
# In[7]:

# import library pandas
import pandas as pd

# membaca file survei_tinggi_badan.txt
df = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/survei_tinggi_badan.txt")

# melakukan rename terhadap kolom "tinggi badan (cm)" menjadi "tinggi"
df.rename({'tinggi badan (cm)' : 'tinggi'}, axis=1, inplace=True)

# mendapatkan nilai mean menggunakan fungsi mean yang disediakan pandas
print(df["tinggi"].mean())

# mendapatkan nilai median menggunakan fungsi median yang disediakan pandas
print(df["tinggi"].median())

# mendapatkan nilai modus menggunakan fungsi mode yang disediakan pandas
print(df["tinggi"].mode())

