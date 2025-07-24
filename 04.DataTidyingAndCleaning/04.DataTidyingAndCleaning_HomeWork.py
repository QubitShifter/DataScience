import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from Utils.functions.func_fix_column_names import snake_case
from Utils.functions.func_convert_units_kg import units_to_kg
from Utils.functions.func_fix_harvest_date import get_start_end_dates
from Utils.functions.func_fix_date_multiColumns import convert_date_multiColumns
from Utils.functions.func_chunk_size import chunk_size
from Utils.functions.func_countryHarvestYear import transform_dates_from_country_and_year
from Utils.dicts.dict_Cofee_Harvest_periods import coffee_harvest_seasons
from Utils.functions.func_HarvestDate_normalized import normalize_harvest_year
from Utils.dicts.dict_Country_on_Continent import country_to_continent
from Utils.lists.stats_Raiting_Columns import statistical_columns


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.precision', 2)
coffee_data = pd.read_csv("lab/merged_data_cleaned.csv")



## 1 Reading DataSet (reading columns, types, etc)
print(f"{'':48}")
print("------------begin dataset observation----------")
print(f"{'':48}\n" * 3)
print(coffee_data.shape)
print(f"{'':48}")
print(coffee_data)
print(coffee_data.columns)
print(coffee_data.dtypes)
print(f"{'':48}\n" * 3)
print("Initial Columns:\n")
print(f"--------------------------------------")
print(f"--------------------------------------")
cols = list(coffee_data.columns)
for i in range(0, len(cols), 5):
    print(cols[i:i+5])
print(f"--------------------------------------")
print(f"{'':48}")
print(f"--------------------------------------")

print("\nData Types:\n", coffee_data.dtypes)
print(f"--------------------------------------")
print(f"{'':48}")
print(f"--------------------------------------")

print("\nFirst few rows:\n", coffee_data.head())
print(f"{'':48}")
print("------------end of reading DataSet----------")
print(f"{'':48}\n" * 3)

## 2 DataSet observation, features, numerical and categorical, etc
print("-------------------------------------------------")
print("-------------begin DataSet observation-----------")
print(f"{'':48}\n" * 1)
print(f"\nNumber of observations: {coffee_data.shape[0]}")
print(f"Number of features: {coffee_data.shape[1]}")

numerical = coffee_data.select_dtypes(include=[np.number]).columns.tolist()
categorical = coffee_data.select_dtypes(include='object').columns.tolist()

if 'Unnamed: 0' in numerical:
    numerical.remove('Unnamed: 0')

print(f"{'':48}\n" * 1)
print(f"\nNumber of Numerical features: {len(numerical)}")
print("\nNumerical features:\n", numerical)
print(f"\nNumber of Numerical features: {len(categorical)}")
print("\nCategorical features:\n", categorical)

print("-------------end of DataSet observation-----------")
print(f"{'':48}")
#
## 3 Column manipulation
###calling function to fix column names to snake_case
print('------------------------------------------------')
print("----------------begin column rename-------------")
print(f"{'':48}\n" * 3)
print("Initial Columns before renaming:\n")
coffee_data_columns_renamed = coffee_data.columns.map(snake_case)
coffee_data_columns_original_names = coffee_data.columns.copy()
coffee_data.columns = coffee_data_columns_renamed
print(coffee_data_columns_original_names)
print(f"{'':48}\n" * 3)
print("DataSet Columns after being renamed:\n")
print(coffee_data.columns)
coffee_data.set_index('unnamed: 0', inplace=True)
print(coffee_data.index.name)
print("---------------end of column rename-------------")
print('------------------------------------------------')
print(f"{'':48}\n" * 2)
#
## 4 Bag Weight
###calling function to fix column names to snake_case
print('------------------------------------------------')
print("----------------bag column ---------------------")
print(f"{'':48}\n" * 1)
print(coffee_data["bag_weight"])
print(f"{'':48}\n" * 1)
print("Initial values in [bag_weight] :\n")
print(coffee_data["bag_weight"].unique())
print(coffee_data["bag_weight"].isna())
print(coffee_data["bag_weight"].isna().any())

coffee_data["is_missing"] = coffee_data["bag_weight"].isna()
print(coffee_data[["bag_weight", 'is_missing']])
print(coffee_data["bag_weight"].isna().sum())
print('------------------------------------------------')
# for value, missing in zip(coffee_data["Bag.Weight"], coffee_data["Bag.Weight"].isna()):
#     print(f"{value} -> Missing: {missing}")

#Bag.Weight column contains as many different measurment units as you can possibly think of.
#Column should be normalized with a single unit, most likely kilogram as ubiversal SI unit for measuring weight.


coffee_data["bag_weight [kg]"] = coffee_data["bag_weight"].apply(units_to_kg)
print(coffee_data[["bag_weight", "bag_weight [kg]"]].head(10))
print(coffee_data["bag_weight [kg]"].unique())

print("---------------end of column rename-------------")
print('------------------------------------------------')
print(' ' * 48)
#
# ## 5 Harvester of Sorrow
# ###deal with it! YOU MUST
#
# #### converting dates in ["grading_date", "expiration"] columns to something easier to work with -> %Y-%m-%d
# print(coffee_data[["grading_date", "expiration"]])
#
# coffee_data_expiration_dates = convert_date_multiColumns(coffee_data, ["grading_date", "expiration"])
# print(coffee_data_expiration_dates[["grading_date", "expiration"]])
# print(coffee_data_expiration_dates[["grading_date", "expiration"]].isna().any())
#
# ### Harvest year
# print("Harvest Year")
# print(coffee_data["harvest_year"].unique())
#
#
# coffee_suppliers = (
#     coffee_data[["country_of_origin", "region", "harvest_year"]]
#     .dropna()
#     .drop_duplicates()
#     .reset_index(drop=True)
#
#
# )
#
# print(chunk_size(coffee_suppliers, chunk_size=20))
#
# #coffee_data[["start_date", "end_date"]] = coffee_data['harvest_year'].apply(lambda x: pd.Series(get_start_end_dates(x)))
# coffee_data[['start_date', 'end_date']] = coffee_data.apply(
#     lambda row: pd.Series(normalize_harvest_year(
#         row['country_of_origin'], row['harvest_year'], coffee_harvest_seasons
#     )),
#     axis=1
# )
#
#
# coffee_data['start_date'] = pd.to_datetime(coffee_data['start_date'], errors='coerce')
# coffee_data['end_date'] = pd.to_datetime(coffee_data['end_date'], errors='coerce')
#
# print('---------------------output by Country name---------------------------------')
# print(
#     coffee_data[coffee_data["country_of_origin"].str.lower() == 'ethiopia']
#     [["country_of_origin", "region", 'harvest_year', 'start_date', 'end_date']]
# )
# print('---------------------Sample preview---------------------------')
# print(
#     coffee_data[["country_of_origin", "region", 'harvest_year', 'start_date', 'end_date']].head()
# )
#
# print('---------------------Missing parsed dates---------------------')
# print(
#     coffee_data[coffee_data['start_date'].isna()]
#     [["country_of_origin", "harvest_year", "region", "start_date", "end_date"]]
# )
#
#
# print('---------------------Fallback to known harvest season---------------------')
# mex_fallback = coffee_data[
#     (coffee_data['country_of_origin'] == 'Mexico') &
#     (coffee_data['start_date'].dt.month == 11)  # November is common start
# ]
# print(mex_fallback[['harvest_year', 'start_date', 'end_date']])
#
#
# #I have newer believed that i will see something more messy than the school backpack of my 8th years old boy,
# # but hey, here we go.. this column hit me hard...
#
#
#
#
# cols = list(coffee_data.columns)
# for i in range(0, len(cols), 5):
#     print(cols[i:i+5])
#
# print(coffee_data["Bag.Weight"])
#
# print("--------------")
#
# print("--------------")
# print(coffee_data["Bag.Weight"].isna().any())
#
# print(coffee_data["Bag.Weight"].apply(lambda x: f"{x} -> Missing: {pd.isna(x)}"))
#
#
# print(coffee_data.columns)
# print("-----------------")
#
# print(coffee_data["Country.of.Origin"].unique())
# print("-----------------")
#
# print(coffee_data["Bag.Weight"].unique())



#
#
#
#
# '''''''''
#
# ## 6 Countries
# ###How many coffees are there with unknown countries of origin? What can you do about them?
#
# # Normalize to lowercase strings and strip whitespace
# countries = coffee_data['country_of_origin'].astype(str).str.strip().str.lower()
# print(countries)
#
# # Define unknown markers
# unknown_markers = ['', 'unknown', 'n/a', 'unk', 'none']
# unknown_countries = countries.isin(unknown_markers) | coffee_data['country_of_origin'].isna()
#
# totalUnknownCount = unknown_countries.sum()
# print(f"Number of coffee with unknown countries of origin is: {totalUnknownCount}")
#
# print(coffee_data[unknown_countries][['region', 'producer', 'mill']].head(10))
# print("---------finding more info for nan's")
# unknown_rows = coffee_data[unknown_countries]
# print(unknown_rows[['region', 'producer', 'mill', 'farm_name', 'company']].head(15))
# print("here we go")
# print(coffee_data.loc[1197])
# #### According to information that we have gather, we can aither drop tjat line, or we can try to populate 'country_of_origin' based
# #### on the information for racafe & cia s.c.a country of production
# #### printing all columns to find some meaningful information. We can find some info in owner and  owner_1 columns we can see 'Racafe & Cia S.C.A'
# #### according to Internet https://drwakefield.com/producer/racafe/ is Columbian coffee brand


#
## 7 Owners
###There are two suspicious columns, named Owner, and Owner.1 (they're likely called something different after you solved problem 3).
### Do something about them. Is there any link to Producer?

### comparing both 'owner' and 'owner_1'

# owners_column_diff = coffee_data[['country_of_origin', 'region', 'producer', 'aroma', 'flavor', 'owner', 'owner_1']]
# diff_owners = owners_column_diff[owners_column_diff['owner'] != owners_column_diff['owner_1']]
# print("column coparison of 'owner' and 'owner_1' differ:\n")
# print(diff_owners.head(15))
#
# same_owners = (coffee_data['owner'] == coffee_data['owner_1']).all()
# print(f"all 'owner' and 'owner_1' always equal? {same_owners}")
#
# count_diff = (coffee_data['owner'] != coffee_data['owner_1']).sum()
# print(f"Total number of different records in 'owner' and 'owner_1' are {count_diff} ")

##all 'owner' and 'owner_1' always equal? False
##Total number of different records in 'owner' and 'owner_1' are 1336

## checking to see if the diffferences are based only on case sensitive style

# normalize_differences = (
#     coffee_data['owner'].str.strip().str.lower() != coffee_data['owner_1'].str.strip().str.lower()
# )
#
# total_count_case_insesitive = normalize_differences
# print(f"Case insensitive difference is: {total_count_case_insesitive}")
#
# ### Case insensitive difference is: unnamed: 0
# ### based on this we can conclude that both columns contain same information and thus  'owner_1' zan be dropped
#
#
# mismatched_normalized = coffee_data[normalize_differences][
#     ['country_of_origin', 'region', 'producer', 'owner', 'owner_1']
# ]
#
# print(mismatched_normalized.head(10))

### differences are only acse sensitive stype and spaces
### column will be dropped
#
# coffee_data.drop(columns=['owner_1'], inplace=True)
# print(coffee_data.columns)
#
#
# ## 8. Coffee color by country and continent
# ###Create a table which shows how many coffees of each color are there in every country. Leave the missing values as they are.
# ###do the same for continents. You know what continent each country is located in.
#
#
#
# country_color_table = (
#     coffee_data.groupby(['country_of_origin', 'color'])
#     .size()
#     .reset_index(name='coffee_count')
#     .sort_values(by=['country_of_origin', 'coffee_count'], ascending=[True, False])
# )
# print(country_color_table)
#
# ### for the continent tables we have created new dictionary -> 'dict_Country_on_Continent'
# coffee_data['continent'] = coffee_data['country_of_origin'].map(country_to_continent)
#
#
# continent_color_table = (
#     coffee_data.groupby(['continent', 'color'])
#     .size()
#     .reset_index(name='coffee_count')
#     .sort_values(by=['continent','coffee_count'], ascending=[True, False])
# )
# print(continent_color_table)
#
# ## 9. Ratings
# ###The columns Aroma, Flavor, etc., up to Moisture represent subjective ratings.
# ### Explore them. Show the means and range; draw histograms and / or boxplots as needed. You can even try correlations if you want. What's up with all those ratings?
#
# columns_data_type = [
#     'aroma', 'flavor', 'aftertaste', 'acidity', 'body', 'balance',
#     'uniformity', 'clean_cup', 'sweetness', 'cupper_points',
#     'total_cup_points', 'moisture'
# ]
#
# print(coffee_data[columns_data_type].dtypes)
#
#
# for col in columns_data_type:
#     valid_data = coffee_data_expiration_dates[col].dropna().iloc[0] if not coffee_data[col].dropna().empty else None
#
# print(f"{col}: {valid_data}")
#
# #### Mean, Min, Max, Range
#
# # stat_desc = coffee_data[statistical_columns].describe().T
# # stat_desc['range'] = stat_desc['max'] - stat_desc['min']
# # print(stat_desc[['mean', 'min', 'max', 'range']])
# #
# # for column in statistical_columns:
# #     plt.figsize = (14, 8)
# #     plt.hist(coffee_data[column].dropna(), bins=20, color='skyblue', edgecolor='black')
# #     plt.suptitle("Coffee Quality Ratings Distribution")
# #     plt.xlabel('Rating')
# #     plt.ylabel('Frequency')
# #     plt.grid(True)
# #     plt.tight_layout()
# #     plt.subplots_adjust()
# #     plt.show()
#
#
# score_counts = coffee_data['aroma'].dropna().round(1).value_counts().sort_index()
#
# plt.figure(figsize=(10, 5))
# score_counts.plot(kind='bar', color='skyblue')
# plt.title("Frequency of Aroma Scores")
# plt.xlabel("Aroma Score")
# plt.ylabel("Number of Coffees")
# plt.xticks(rotation=45)
# plt.grid(axis='y')
# plt.tight_layout()
# plt.show()
#
#
#
# columns_to_analyze = [
#     'aroma', 'flavor', 'aftertaste', 'acidity', 'body', 'balance',
#     'uniformity', 'clean_cup', 'sweetness', 'cupper_points',
#     'total_cup_points', 'moisture'
# ]
#
# # Set up subplot grid: 3 rows x 4 columns
# fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 7))
# axes = axes.flatten()  # Flatten to iterate easily
#
# # Loop through each column and plot histogram
# for i, col in enumerate(columns_to_analyze):
#     axes[i].hist(coffee_data[col].dropna(), bins=20, color='skyblue', edgecolor='black')
#     axes[i].set_title(f"{col.capitalize()} Distribution")
#     axes[i].set_xlabel(col, labelpad=10)
#     axes[i].set_ylabel("Frequency")
#     axes[i].grid(True)
#
# # Remove any unused subplots if columns < 12
# for j in range(i + 1, len(axes)):
#     fig.delaxes(axes[j])
#
# plt.tight_layout()
# plt.show()
#
#
# correlation_matrix = coffee_data[statistical_columns].corr()
# print(correlation_matrix)
#
#
# plt.figure(figsize=(12, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=1, vmax=1, fmt=".2f")
# plt.title("Correlation Between COfee Quality Ratings")
# plt.xticks(rotation=45)
# plt.yticks(rotation=0)
# plt.tight_layout()
# plt.show()
#
#
# ratings_clean = coffee_data[statistical_columns].dropna()
#
# # Set up the plot
# plt.figure(figsize=(14, 8))
# sns.boxplot(data=ratings_clean, orient='h', palette='Set2')
# plt.title('Boxplots of Coffee Rating Features')
# plt.xlabel('Score')
# plt.ylabel('Feature')
# plt.grid(axis='x', linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()
#
#
# ratings_clean = coffee_data[statistical_columns].dropna()
#
# # Melt the DataFrame for seaborn
# melted = ratings_clean.melt(var_name='Feature', value_name='Score')
#
# # Plot
# plt.figure(figsize=(14, 6))
# sns.boxplot(x='Feature', y='Score', data=melted, palette='Set2')
# plt.title('Boxplots of Coffee Rating Features')
# plt.xticks(rotation=45)
# plt.xlabel('Feature')
# plt.ylabel('Score')
# plt.ylim(5, 90)
# plt.grid(axis='y', linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()
#
#
# scaled_data = coffee_data[statistical_columns].copy()
#
# # Apply MinMax scaling to 0–10 range
# scaler = MinMaxScaler(feature_range=(0, 10))
# scaled_data[columns_to_analyze] = scaler.fit_transform(scaled_data[columns_to_analyze])
#
# # Melt for seaborn
# melted_scaled = scaled_data.melt(var_name='Feature', value_name='Score')
#
# # Plot boxplot
# plt.figure(figsize=(14, 6))
# sns.boxplot(x='Feature', y='Score', data=melted_scaled, palette='Set3')
# plt.title('Boxplots of Coffee Ratings (Normalized)')
# plt.xticks(rotation=45)
# plt.ylabel('Normalized Score (0–10)')
# plt.grid(axis='y', linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()
#
# ##10. High-level errors
# ### Check the countries against region names, altitudes, and companies.
# ### Are there any discrepancies (e.g. human errors, like a region not matching the country)?
# ### Take a look at the (cleaned) altitudes; there has been a lot of preprocessing done to them. Was it done correctly?
#
# #### проверяваме страна сръщу регион
#
# country_region = coffee_data.groupby('country_of_origin')['region'].unique()
# for country, regions in country_region.items():
#     print(f"{country}: {regions}")
#
#
# #### check altitudes
#
# altitude_check = coffee_data[
#     (coffee_data['altitude_mean_meters'] < coffee_data['altitude_low_meters']) |
#     (coffee_data['altitude_mean_meters'] > coffee_data['altitude_high_meters'])
# ]
#
# print("Rows where mean is outside low-high range:")
# print(altitude_check[['country_of_origin', 'region', 'altitude_low_meters', 'altitude_high_meters', 'altitude_mean_meters']])
#
#
# #### this prints:
# '''''
# Rows where mean is outside low-high range:
# Empty DataFrame
# Columns: [country_of_origin, region, altitude_low_meters, altitude_high_meters, altitude_mean_meters]
# Index: []
# '''''
#
# #### which implies that the altitude cleaning and normalization appears to be correct.
# #### no obvious discrepancies (like a mean outside the min/max range) were found.
# #### the data seems internally consistent for altitudes.
#
# print(coffee_data[['altitude_low_meters', 'altitude_high_meters', 'altitude_mean_meters']].isnull().sum())
#
# same_altitude = coffee_data[
#     (coffee_data['altitude_low_meters'] == coffee_data['altitude_high_meters']) &
#     (coffee_data['altitude_low_meters'] == coffee_data['altitude_mean_meters'])
# ]
#
# print(f"Rows with all altitude values equal: {len(same_altitude)}")
# print(same_altitude[['country_of_origin', 'region', 'altitude_low_meters']].value_counts().head(10))
#
#
#
# # Step 1: Create the 'altitude_all_equal' column
# coffee_data['altitude_all_equal'] = (
#     (coffee_data['altitude_low_meters'] == coffee_data['altitude_high_meters']) &
#     (coffee_data['altitude_low_meters'] == coffee_data['altitude_mean_meters'])
# )
#
#
# implausible_altitudes = coffee_data[
#     (coffee_data['altitude_all_equal']) & (coffee_data['altitude_mean_meters'] < 100)
# ]
# print(implausible_altitudes[['country_of_origin', 'region', 'altitude_mean_meters']])
#
#
# #### there are some not realistic altitudes for South America  mountain countries