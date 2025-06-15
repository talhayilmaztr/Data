import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2

# Reading the CSV file
df = pd.read_csv("Data/smoking_data2.csv")

# Dropping missing values in the 'cigs_per_day' column
cigs = df["cigs_per_day"].dropna()

# Getting the number of valid data
n = len(cigs)

# Calculating mean, median, variance, standard deviation, and standard error
mean = cigs.mean()
median = cigs.median()
var = cigs.var()
std = cigs.std()
se = std / math.sqrt(n)

# Printing the results
print("Mean:", round(mean, 2))
print("Median:", round(median, 2))
print("Variance:", round(var, 2))
print("Standard Deviation:", round(std, 2))
print("Standard Error:", round(se, 4))

# z-score for 95% confidence interval
z = 1.96

# Calculating confidence interval for the mean
mean_low = mean - z * se
mean_high = mean + z * se

# Calculating confidence interval for variance (using chi-square distribution)
dfree = n - 1
chi_low = chi2.ppf(0.025, dfree)
chi_high = chi2.ppf(0.975, dfree)
var_low = (dfree * var) / chi_high
var_high = (dfree * var) / chi_low

# Printing confidence intervals
print("\n95% Confidence Interval (Mean):", round(mean_low, 2), "-", round(mean_high, 2))
print("95% Confidence Interval (Variance):", round(var_low, 2), "-", round(var_high, 2))

# Hypothesis test: is the mean really 10?
mu_0 = 10
z_score = (mean - mu_0) / se
p = 2 * (1 - norm.cdf(abs(z_score)))  # two-tailed test

# Printing test results
print("\nZ Score:", round(z_score, 2))
print("P Value:", round(p, 4))

# Sample size calculation (±1 margin of error, 90% confidence level)
z90 = norm.ppf(0.95)
E = 1
n_gerekli = ((z90 * std) / E) ** 2

# Printing required sample size
print("\nRequired Sample Size:", math.ceil(n_gerekli))

# Plotting histogram
plt.hist(cigs, bins=20, color="lightblue", edgecolor="black")
plt.title("Daily Cigarette Consumption (Histogram)")
plt.xlabel("Number of Cigarettes")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Plotting boxplot
plt.boxplot(cigs, patch_artist=True)
plt.title("Daily Cigarette Consumption (Boxplot)")
plt.ylabel("Number of Cigarettes")
plt.grid(True)
plt.show()
