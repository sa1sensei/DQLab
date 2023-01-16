-- Praktikum
SELECT DISTINCT
	Customer_ID, Product,
    Average_transaction,
    Average_transaction >= 1000000 is_eligible
FROM summary_transaction;

-- Logika AND
SELECT
   Customer_ID, Product, average_transaction_amount,
   product = 'Jaket' AND average_transaction_amount >= 1000000 loyal_buyer_jaket
FROM data_retail
WHERE product = 'Jaket'


-- Logika OR
SELECT 
     Customer_ID, 
     Product, 
     average_transaction_amount,
     product = 'Jaket' OR product = 'Baju' buyer_fashion
FROM data_retail;


-- Logika NOT
SELECT 
       *
FROM data_retail
WHERE NOT product = 'Jaket';

-- Logika XOR
SELECT 
    Customer_ID,
    Product,
    Average_Transaction_Amount,
    product = 'Jaket' XOR average_transaction_amount > 1000000 logika_xor
FROM data_retail;


-- Pemanfaatan Operator Logika dalam Perintah Where
SELECT DISTINCT
     *
FROM data_retail
WHERE Product = "Jaket" AND Average_Transaction_Amount >= 1000000;

-- Pemanfaatan Operator Logika dalam Konstruksi Case When
SELECT DISTINCT
Customer_ID,
Product,
Average_Transaction,
Count_Transaction,
CASE
     WHEN Average_transaction > 2000000 and Count_Transaction > 30 then 'Platinum Member'
     WHEN Average_transaction between 1000000 and 2000000 and Count_Transaction between 20 and 30 then 'Gold Member'
     WHEN Average_transaction < 1000000 and Count_Transaction<20 then 'Silver Member'
         ELSE "Other Membership" end as Membership
FROM summary_transaction


-- Praktikum
SELECT DISTINCT
Customer_ID,
Product,
Average_transaction,
Count_Transaction,
CASE
	WHEN Average_transaction <1000000 then '4-5x Email dalam seminggu'
    WHEN Average_transaction >1000000 then '1-2x Email dalam seminggu'
END AS frekuensi_email
FROM summary_transaction;


-- Perintah Where dan Logika And
SELECT DISTINCT
Customer_ID
FROM summary_transaction
WHERE Average_transaction < 1000000 and product =  'Sepatu';


-- Menyiapkan Report Penjualan
SELECT DISTINCT
Product Produk,
avg(Average_transaction) 'Jumlah transaksi (Rupiah)',
sum(Count_Transaction) 'Barang terjual'
FROM summary_transaction
GROUP BY Product

-- Tugas Pertamaku
SELECT DISTINCT
	Customer_ID
FROM data_retail_undian
WHERE Unik_produk > 3 AND Rata_rata_transaksi > 1500000


-- Tugas Keduaku
SELECT DISTINCT
	Customer_ID
FROM data_retail_undian
WHERE NOT (Unik_produk >= 3 AND Rata_rata_transaksi >= 1500000)

