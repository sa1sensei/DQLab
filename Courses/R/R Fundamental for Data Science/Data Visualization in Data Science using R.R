################# Membuat Kanvas Kosong

library(ggplot2)

# Ketik function ggplot() di bawah ini
ggplot()




################# Menambahkan Judul

library(ggplot2)

# Penambahan judul dengan menggunakan fungsi labs
ggplot() + labs(title = "Luas Wilayah vs Kepadatan Penduduk DKI Jakarta - Periode 2013")




################# Plot disimpan sebagai Variable

library(ggplot2)

plot.jakarta <- ggplot()
plot.jakarta <- plot.jakarta + labs(title="Luas Wilayah vs Kepadatan Penduduk DKI Jakarta - Periode 2013")
plot.jakarta




################# Menambahkan Label pada Sumbu X dan Y

library(ggplot2)
plot.jakarta <- ggplot()
plot.jakarta <- plot.jakarta + labs(title="Luas Wilayah vs Kepadatan Penduduk DKI Jakarta - Periode 2013", subtitle="Tahun 2013")
plot.jakarta <- plot.jakarta labs(x="Luas Wilayah (km2)", y = "Kepadatan Jiwa per km2")
plot.jakarta





################# Fungsi summary untuk objek ggplot

library(ggplot2)

plot.jakarta <- ggplot()
plot.jakarta <- plot.jakarta + labs(title="Luas Wilayah vs Kepadatan Penduduk DKI Jakarta")
plot.jakarta <- plot.jakarta + labs(x = "Luas Wilayah (km2)", y="Kepadatan Jiwa per km2")
summary(plot.jakarta)





################# Membaca Dataset Kependudukan dengan read.csv

library(ggplot2)
#Membaca data csv dan dimasukkan ke variable penduduk.dki
penduduk.dki <- read.csv("https://storage.googleapis.com/dqlab-dataset/dkikepadatankelurahan2013.csv", sep=",")
# Tampilkan data frame dari kolom " NAMA.KELURAHAN " dan "LUAS.WILAYAH..KM2."
penduduk.dki[c("NAMA.KELURAHAN", "LUAS.WILAYAH..KM2.")]




################# Memasukkan Data ke Plot

library(ggplot2)
#Membaca data csv dan dimasukkan ke variable penduduk.dki
penduduk.dki <- read.csv("https://storage.googleapis.com/dqlab-dataset/dkikepadatankelurahan2013.csv", sep=",")
# Masukkan data ke dalam plot dan simpan sebagai variable plot.dki, dan tampilkan summary dari plot tersebut
plot.dki <- ggplot(data = penduduk.dki)
summary(plot.dki)




################# Memetakan x, y dan color dengan function aes

library(ggplot2)
#Membaca data csv dan dimasukkan ke variable penduduk.dki
penduduk.dki <- read.csv("https://storage.googleapis.com/dqlab-dataset/dkikepadatankelurahan2013.csv", sep=",")
plot.dki <- ggplot(data=penduduk.dki, aes(x = LUAS.WILAYAH..KM2.,  y=KEPADATAN..JIWA.KM2., color=NAMA.KABUPATEN.KOTA))
summary(plot.dki)




################# Menampilkan Plot hasil Mapping

library(ggplot2)
#Membaca data csv dan dimasukkan ke variable penduduk.dki
penduduk.dki <- read.csv("https://storage.googleapis.com/dqlab-dataset/dkikepadatankelurahan2013.csv", sep=",")
plot.dki <- ggplot(data=penduduk.dki, aes(x = LUAS.WILAYAH..KM2.,  y=KEPADATAN..JIWA.KM2.,  color=NAMA.KABUPATEN.KOTA))
plot.dki





################# Scatter Plot Kepadatan Penduduk Jakarta dengan function layer

library(ggplot2)
#Membaca data csv dan dimasukkan ke variable penduduk.dki
penduduk.dki <- read.csv("https://storage.googleapis.com/dqlab-dataset/dkikepadatankelurahan2013.csv", sep=",")

#Menambahkan data dan aesthetic mapping
plot.dki <- ggplot(data=penduduk.dki, aes(x = LUAS.WILAYAH..KM2.,  y=KEPADATAN..JIWA.KM2.,  color=NAMA.KABUPATEN.KOTA))

#Menambahkan layer untuk menghasilkan grafik scatter plot
plot.dki + layer(geom = "point", stat = "identity", position = "identity")




################# Scatter Plot Kepadatan Penduduk Jakarta dengan geom_point

library(ggplot2)
#Membaca data csv dan dimasukkan ke variable penduduk.dki
penduduk.dki <- read.csv("https://storage.googleapis.com/dqlab-dataset/dkikepadatankelurahan2013.csv", sep=",")

#Menambahkan data dan aesthetic mapping
plot.dki <- ggplot(data=penduduk.dki, aes(x = LUAS.WILAYAH..KM2.,  y=KEPADATAN..JIWA.KM2.,  color=NAMA.KABUPATEN.KOTA))

# Menambahkan layer scatter plot dengan geom_point
plot.dki + geom_point()




################# Menambahkan Judul dan Label

library(ggplot2)

#Membaca data csv dan dimasukkan ke variable penduduk.dki
penduduk.dki <- read.csv("https://storage.googleapis.com/dqlab-dataset/dkikepadatankelurahan2013.csv", sep=",")

#Menambahkan data dan aesthetic mapping
plot.dki <- ggplot(data=penduduk.dki, aes(x = LUAS.WILAYAH..KM2.,  y=KEPADATAN..JIWA.KM2.,  color=NAMA.KABUPATEN.KOTA))

# Menambahkan Layer dan labels
plot.dki + geom_point() + 
  theme(plot.title = element_text(hjust=0.5)) +
  labs(title = "Luas Wilayah vs Kepadatan Penduduk DKI Jakarta",
       x = "Luas wilayah (km2)",
       y = "Kepadatan Jiwa per km2",
       color = "Nama Kabupaten/Kota")




################# Layer geom_histogram dan Lebar Interval

library(ggplot2)

#Membaca data csv dan dimasukkan ke variable penduduk.dki
penduduk.dki <- read.csv("https://storage.googleapis.com/dqlab-dataset/dkikepadatankelurahan2013.csv", sep=",")

# Menambahkan data dan aesthetic mapping
plot.dki <- ggplot(data=penduduk.dki, aes(x = KEPADATAN..JIWA.KM2.))
plot.dki + geom_histogram(binwidth=10000)





################# Penggunaaan aesthetic fill

library(ggplot2)

#Membaca data csv dan dimasukkan ke variable penduduk.dki
penduduk.dki <- read.csv("https://storage.googleapis.com/dqlab-dataset/dkikepadatankelurahan2013.csv", sep=",")
plot.dki <- ggplot(data=penduduk.dki, aes(x = KEPADATAN..JIWA.KM2., fill = NAMA.KABUPATEN.KOTA))
plot.dki + geom_histogram(binwidth = 10000)




################# Membaca data inflasi

#Membaca data csv dan dimasukkan ke variable inflasi.indo.sing
inflasi.indo.sing <- read.csv("https://storage.googleapis.com/dqlab-dataset/inflasi.csv", sep=",")
inflasi.indo.sing




################# Plotting Line Chart yang Kosong

library(ggplot2)

#Membaca data csv dan dimasukkan ke variable inflasi.indo.sing
inflasi.indo.sing <- read.csv("https://storage.googleapis.com/dqlab-dataset/inflasi.csv", sep=",")

#Menambahkan data dan aesthetic mapping
plot.inflasi <- ggplot(data=inflasi.indo.sing, aes(x = Bulan,  y=Inflasi,  color=Negara))

#Menambahkan layer
plot.inflasi + geom_line()




################# Menggunakan Pengelompokan Data (grouping)

library(ggplot2)

#Membaca data csv dan dimasukkan ke variable inflasi.indo.sing
inflasi.indo.sing <- read.csv("https://storage.googleapis.com/dqlab-dataset/inflasi.csv", sep=",")

#Menambahkan data dan aesthetic mapping
plot.inflasi <- ggplot(data=inflasi.indo.sing, aes(x=Bulan, y=Inflasi, color=Negara, group=Negara))

#Menambahkan Layer
plot.inflasi + geom_line()





################# Memperbaiki Urutan Bulan dengan Factoring

library(ggplot2)
#Membaca data csv dan dimasukkan ke variable inflasi.indo.sing
inflasi.indo.sing <- read.csv("https://storage.googleapis.com/dqlab-dataset/inflasi.csv", sep=",")
inflasi.indo.sing$Bulan <- factor(inflasi.indo.sing$Bulan, 
                                  levels = c("Jan-2017", "Feb-2017", "Mar-2017", "Apr-2017", "May-2017", 
                                             "Jun-2017", "Jul-2017", "Aug-2017", "Sep-2017", "Oct-2017"))

str(inflasi.indo.sing)




################# Plotting Ulang dengan hasil Factoring

library(ggplot2)

#Membaca data csv dan dimasukkan ke variable inflasi.indo.sing
inflasi.indo.sing <- read.csv("https://storage.googleapis.com/dqlab-dataset/inflasi.csv", sep=",")

inflasi.indo.sing$Bulan = factor(inflasi.indo.sing$Bulan, 
                                 levels = c("Jan-2017", "Feb-2017", "Mar-2017", "Apr-2017", "May-2017", "Jun-2017", "Jul-2017", "Aug-2017", "Sep-2017", "Oct-2017"))

#Menambahkan data dan aesthetic mapping
plot.inflasi <- ggplot(data=inflasi.indo.sing, aes(x = Bulan,  y=Inflasi,  color=Negara, group=Negara))


#Menambahkan Layer dan labels
plot.inflasi + geom_line() + geom_text(aes(label=Inflasi),hjust=-0.2, vjust=-0.5)





################# Menghasilkan Bar Chart Pertama

library(ggplot2)
#Membaca data csv dan dimasukkan ke variable penduduk.dki
penduduk.dki <- read.csv("https://storage.googleapis.com/dqlab-dataset/datakependudukandki-dqlab.csv", sep=",")
plot.dki <- ggplot(data=penduduk.dki, aes(x = NAMA.KABUPATEN.KOTA))
plot.dki + geom_bar()




################# Aesthetic Y dan Stat Identity

library(ggplot2)

#Membaca data csv dan dimasukkan ke variable penduduk.dki
penduduk.dki <- read.csv("https://storage.googleapis.com/dqlab-dataset/datakependudukandki-dqlab.csv", sep=",")

#Menghasilkan bar chart

# Membuat plot
plot.dki <- ggplot(data=penduduk.dki, aes(x = NAMA.KABUPATEN.KOTA, y = JUMLAH))

# Menambahkan layer pada plot
plot.dki + geom_bar(stat="identity")




################# Aesthetic Fill dan Position Dodge

library(ggplot2)

#Membaca data csv dan dimasukkan ke variable penduduk.dki
penduduk.dki <- read.csv("https://storage.googleapis.com/dqlab-dataset/datakependudukandki-dqlab.csv", sep=",")

# Bagian plot
plot.dki <- ggplot(data=penduduk.dki, aes(x=NAMA.KABUPATEN.KOTA, y=JUMLAH, fill=JENIS.KELAMIN))

# Bagian penambahan layer
plot.dki + geom_bar(stat="identity", position="dodge")




################# Fungsi Aggregate

library(ggplot2)

#Membaca data csv dan dimasukkan ke variable penduduk.dki
penduduk.dki <- read.csv("https://storage.googleapis.com/dqlab-dataset/datakependudukandki-dqlab.csv", sep=",")
# Melakukan agregasi
aggregate(x=list(JUMLAH=penduduk.dki$JUMLAH), 
          FUN=sum, 
          by = list(NAMA.KABUPATEN.KOTA=penduduk.dki$NAMA.KABUPATEN.KOTA, 
                    JENIS.KELAMIN=penduduk.dki$JENIS.KELAMIN))





################# "Merapikan" Tampilan Bar Chart

library(ggplot2)
#Membaca data csv dan dimasukkan ke variable penduduk.dki
penduduk.dki <- read.csv("https://storage.googleapis.com/dqlab-dataset/datakependudukandki-dqlab.csv", sep=",")
#Melakukan agregasi
p <- aggregate(x=list(JUMLAH=penduduk.dki$JUMLAH), FUN=sum, by = list(NAMA.KABUPATEN.KOTA=penduduk.dki$NAMA.KABUPATEN.KOTA, JENIS.KELAMIN=penduduk.dki$JENIS.KELAMIN))

#Plot grafik
plot.dki <- ggplot(data=p, aes(x = NAMA.KABUPATEN.KOTA, y=JUMLAH, fill=JENIS.KELAMIN, label = JUMLAH))
plot.dki <- plot.dki + geom_bar(stat="identity", position="dodge")  
plot.dki <- plot.dki + labs(title="Jumlah Penduduk DKI Jakarta Umur > 35 - Tahun 2013", x="Kabupaten / Kota", y="Jumlah Penduduk")
plot.dki <- plot.dki + theme(axis.text.x = element_text(angle=45,vjust = 0.5), plot.title = element_text(hjust=0.5))
plot.dki <- plot.dki + geom_text(position = position_dodge(1.2))
plot.dki





################# Pie Chart dengan Koordinat Polar

library(ggplot2)

#Membaca data csv dan dimasukkan ke variable penduduk.dki
penduduk.dki <- read.csv("https://storage.googleapis.com/dqlab-dataset/datakependudukandki-dqlab.csv", sep=",")
#Melakukan agregasi
p <- aggregate(x=list(JUMLAH=penduduk.dki$JUMLAH), FUN=sum, by = list(NAMA.KABUPATEN.KOTA=penduduk.dki$NAMA.KABUPATEN.KOTA))
#Plot grafik pie chart
plot.dki <- ggplot(data=p, aes(x="", y=JUMLAH, fill = NAMA.KABUPATEN.KOTA))

plot.dki <- plot.dki + geom_bar(width = 1, stat = "identity")
plot.dki <- plot.dki + coord_polar("y", start=0)
plot.dki




################# Faceting pada Scatter Plot

library(ggplot2)

#Membaca data csv dan dimasukkan ke variable penduduk.dki
penduduk.dki <- read.csv("https://storage.googleapis.com/dqlab-dataset/dkikepadatankelurahan2013.csv", sep=",")

#Menambahkan data dan aesthetic mapping
plot.dki <- ggplot(data=penduduk.dki, aes(x = LUAS.WILAYAH..KM2.,  y=KEPADATAN..JIWA.KM2.,  color=NAMA.KABUPATEN.KOTA))


# Menambahkan layer
plot.dki <- plot.dki + layer(geom = "point", stat="identity", position = "identity")
plot.dki <- plot.dki + labs(x="Luas Wilayah (km2)", y="Kepadatan Jiwa (km2)", color="Kabupaten/Kota")
plot.dki + facet_wrap( ~ NAMA.KABUPATEN.KOTA, ncol=2)





################# Faceting pada Histogram

library(ggplot2)

#Membaca data csv dan dimasukkan ke variable penduduk.dki
penduduk.dki <- read.csv("https://storage.googleapis.com/dqlab-dataset/dkikepadatankelurahan2013.csv", sep=",")

#Menambahkan data dan aesthetic mapping
plot.dki <- ggplot(data=penduduk.dki, aes(x=KEPADATAN..JIWA.KM2.,  fill=NAMA.KABUPATEN.KOTA))

# Menambahkan layer
plot.dki <- plot.dki + geom_histogram(binwidth=10000)
plot.dki <- plot.dki + labs(x="Kepadatan Jiwa (km2)", y="Jumlah Kelurahan", color="Kabupaten/Kota")
plot.dki + facet_wrap( ~ NAMA.KABUPATEN.KOTA, ncol=2)

