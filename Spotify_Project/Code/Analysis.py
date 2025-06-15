import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2

# CSV dosyasından veriyi okuyoruz
df = pd.read_csv("Data/smoking_data2.csv")

# 'cigs_per_day' sütunundaki eksik verileri siliyoruz
cigs = df["cigs_per_day"].dropna()

# Geçerli veri sayısını buluyoruz
n = len(cigs)

# Ortalama, ortanca, varyans, standart sapma ve standart hata hesaplıyoruz
mean = cigs.mean()
median = cigs.median()
var = cigs.var()
std = cigs.std()
se = std / math.sqrt(n)

# Sonuçları yazdırıyoruz
print("Ortalama:", round(mean, 2))
print("Ortanca:", round(median, 2))
print("Varyans:", round(var, 2))
print("Standart Sapma:", round(std, 2))
print("Standart Hata:", round(se, 4))

# %95 güven aralığı için z değeri
z = 1.96

# Ortalama için güven aralığını hesaplıyoruz
mean_low = mean - z * se
mean_high = mean + z * se

# Varyans için güven aralığı (ki-kare dağılımı kullanılarak)
dfree = n - 1
chi_low = chi2.ppf(0.025, dfree)
chi_high = chi2.ppf(0.975, dfree)
var_low = (dfree * var) / chi_high
var_high = (dfree * var) / chi_low

# Güven aralıklarını yazdırıyoruz
print("\n%95 Güven Aralığı (Ortalama):", round(mean_low, 2), "-", round(mean_high, 2))
print("%95 Güven Aralığı (Varyans):", round(var_low, 2), "-", round(var_high, 2))

# Hipotez testi: ortalama gerçekten 10 mu?
mu_0 = 10
z_score = (mean - mu_0) / se
p = 2 * (1 - norm.cdf(abs(z_score)))  # iki yönlü test

# Test sonuçlarını yazdırıyoruz
print("\nZ Skoru:", round(z_score, 2))
print("P Değeri:", round(p, 4))

# Örneklem büyüklüğü hesabı (±1 hata payı, %90 güven düzeyi)
z90 = norm.ppf(0.95)
E = 1
n_gerekli = ((z90 * std) / E) ** 2

# Gerekli örneklem sayısını yazdırıyoruz
print("\nGerekli Örneklem Sayısı:", math.ceil(n_gerekli))

# Histogram çizimi
plt.hist(cigs, bins=20, color="lightblue", edgecolor="black")
plt.title("Günlük Sigara Tüketimi (Histogram)")
plt.xlabel("Sigara Sayısı")
plt.ylabel("Frekans")
plt.grid(True)
plt.show()

# Boxplot (kutucuk grafik) çizimi
plt.boxplot(cigs, patch_artist=True)
plt.title("Günlük Sigara Tüketimi (Boxplot)")
plt.ylabel("Sigara Sayısı")
plt.grid(True)
plt.show()
