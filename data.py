# -----------------------------------------------------------------------------
# Statistics and Probability Term Project - Daily Cigarette Consumption Analysis
# -----------------------------------------------------------------------------

import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2

# Read the dataset
df = pd.read_csv("Data/smoking_data2.csv")  # ← update the path if needed
daily_cigs = df["cigs_per_day"].dropna()
n = len(daily_cigs)

# -----------------------------------------------------------------------------
# Descriptive Statistics
# -----------------------------------------------------------------------------
mean_val = daily_cigs.mean()
median_val = daily_cigs.median()
variance_val = daily_cigs.var()
std_dev_val = daily_cigs.std()
standard_error = std_dev_val / math.sqrt(n)

print("----- Descriptive Statistics -----")
print(f"Mean: {mean_val:.2f}")
print(f"Median: {median_val:.2f}")
print(f"Variance: {variance_val:.2f}")
print(f"Standard Deviation: {std_dev_val:.2f}")
print(f"Standard Error: {standard_error:.4f}")

# -----------------------------------------------------------------------------
# 95% Confidence Intervals
# -----------------------------------------------------------------------------
z = 1.96
ci_mean_lower = mean_val - z * standard_error
ci_mean_upper = mean_val + z * standard_error

alpha = 0.05
dfree = n - 1
chi2_lower = chi2.ppf(alpha/2, dfree)
chi2_upper = chi2.ppf(1 - alpha/2, dfree)
ci_var_lower = (dfree * variance_val) / chi2_upper
ci_var_upper = (dfree * variance_val) / chi2_lower

print("\n----- 95% Confidence Intervals -----")
print(f"Mean: {ci_mean_lower:.2f} - {ci_mean_upper:.2f}")
print(f"Variance: {ci_var_lower:.2f} - {ci_var_upper:.2f}")

# -----------------------------------------------------------------------------
# Hypothesis Test (Is the mean equal to 10?)<<<<<
# -----------------------------------------------------------------------------
mu_0 = 10
z_score = (mean_val - mu_0) / standard_error
p_value = 2 * (1 - norm.cdf(abs(z_score)))

print("\n----- Hypothesis Test (Mean = 10?) -----")
print(f"Z-score: {z_score:.2f}")
print(f"P-value: {p_value:.4f}")

# -----------------------------------------------------------------------------
# Sample Size Estimation (90% confidence, ±1 margin of error)
# -----------------------------------------------------------------------------
E = 1
z_90 = norm.ppf(0.95)
n_required = ((z_90 * std_dev_val) / E) ** 2

print("\n----- Sample Size Estimation -----")
print(f"Required sample size: {math.ceil(n_required)}")

# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
plt.figure(figsize=(10,6))
plt.hist(daily_cigs, bins=20, color='skyblue', edgecolor='black')
plt.title('Daily Cigarette Consumption Distribution (Histogram)')
plt.xlabel('Number of Cigarettes per Day')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

plt.figure(figsize=(6,8))
plt.boxplot(daily_cigs, vert=True, patch_artist=True, showfliers=True)
plt.title('Daily Cigarette Consumption (Boxplot)')
plt.ylabel('Number of Cigarettes')
plt.grid(True)
plt.show()
