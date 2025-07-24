import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Task 1 – Initial Data Load and Observation
# -----------------------------
df = pd.read_csv('lab/merged_data_cleaned.csv')
print("Initial Columns:\n", df.columns)
print("\nData Types:\n", df.dtypes)
print("\nFirst few rows:\n", df.head())

# -----------------------------
# Task 2 – Observations and Features
# -----------------------------
print(f"\nNumber of observations: {df.shape[0]}")
print(f"Number of features: {df.shape[1]}")

numerical = df.select_dtypes(include=[np.number]).columns.tolist()
categorical = df.select_dtypes(include='object').columns.tolist()

print("\nNumerical features:\n", numerical)
print("\nCategorical features:\n", categorical)

# -----------------------------
# Task 3 – Column Name Cleanup to snake_case
# -----------------------------
def to_snake_case(col):
    return col.strip().replace('.', '_').replace(' ', '_').lower()

df.columns = [to_snake_case(col) for col in df.columns]
print("\nRenamed columns:\n", df.columns)

# -----------------------------
# Task 4 – Bag Weight Normalization
# -----------------------------
df['bag_weight_cleaned'] = df['bag_weight'].str.extract(r'(\d+\.?\d*)').astype(float)
# Assuming all weights are in kg
print("\nBag weights (cleaned):\n", df[['bag_weight', 'bag_weight_cleaned']].head())

# -----------------------------
# Task 5 – Date Cleanup
# -----------------------------
# Harvest year might be messy, parse only year digits
df['harvest_year_cleaned'] = df['harvest_year'].str.extract(r'(\d{4})')
df['harvest_year_cleaned'] = pd.to_numeric(df['harvest_year_cleaned'], errors='coerce')

# Convert grading and expiration dates
df['grading_date'] = pd.to_datetime(df['grading_date'], errors='coerce')
df['expiration'] = pd.to_datetime(df['expiration'], errors='coerce')

print("\nSample cleaned dates:\n", df[['grading_date', 'expiration', 'harvest_year_cleaned']].head())

# -----------------------------
# Task 6 – Unknown Countries
# -----------------------------
unknown_countries = df['country_of_origin'].isna().sum()
print(f"\nUnknown countries of origin: {unknown_countries}")

# Optionally fill with known producer country if possible (not shown here due to complexity)

# -----------------------------
# Task 7 – Owner Columns Cleanup
# -----------------------------
# Compare and unify owner columns
df['owner_match'] = df['owner'] == df['owner_1']
print("\nOwner mismatch count:", (~df['owner_match']).sum())

# Keep the more complete one
df['final_owner'] = df['owner_1'].combine_first(df['owner'])

# -----------------------------
# Task 8 – Coffee Color by Country
# -----------------------------
color_table = pd.crosstab(df['country_of_origin'], df['color'])
print("\nCoffee color table:\n", color_table)

# -----------------------------
# Task 9 – Ratings Exploration
# -----------------------------
ratings_cols = [
    'aroma', 'flavor', 'aftertaste', 'acidity', 'body', 'balance',
    'uniformity', 'clean_cup', 'sweetness', 'cupper_points',
    'total_cup_points', 'moisture'
]

ratings_stats = df[ratings_cols].describe().T[['mean', 'min', 'max']]
ratings_stats['range'] = ratings_stats['max'] - ratings_stats['min']
print("\nRatings summary:\n", ratings_stats)

# Plot histograms
df[ratings_cols].hist(figsize=(15, 10))
plt.suptitle("Ratings Histograms")
plt.tight_layout()
plt.show()

# Plot boxplots (flipped)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[ratings_cols], orient='h')
plt.title("Boxplots of Coffee Ratings")
plt.xlabel("Score")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df[ratings_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Between Ratings")
plt.show()

# -----------------------------
# Task 10 – High-Level Errors
# -----------------------------
# Check altitude logic
df['altitude_all_equal'] = (
    df['altitude_low_meters'] == df['altitude_high_meters']
) & (df['altitude_low_meters'] == df['altitude_mean_meters'])

implausible = df[(df['altitude_all_equal']) & (df['altitude_mean_meters'] < 100)]
print(f"\nImplausible altitudes (low): {len(implausible)}")
print(implausible[['country_of_origin', 'region', 'altitude_mean_meters']])

# -----------------------------
# Task 11 – Free Exploration Ideas
# -----------------------------
# Example: Distribution of total cup points by country
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='country_of_origin', y='total_cup_points')
plt.xticks(rotation=90)
plt.title("Total Cup Points by Country")
plt.tight_layout()
plt.show()

# Save cleaned version
a = df.to_csv('coffee_cleaned_final.csv', index=False)
print(a)
