import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from Utils.Functions.func_helper_print_colors import color_print
from Utils.Functions.func_helper_query_dataset import query_dataframe
from Utils.Functions.func_fix_column_names import snake_case
from Utils.Functions.func_fix_date_format import fix_dates
from Utils.Functions.func_helper_remove_commas import remove_commas
from Utils.Functions.func_helper_query_dataset import query_dataframe
from Utils.Functions.func_read_csv_zipped import read_zipped_csv
from Utils.Functions.func_get_encoding import encoding_pre_check


indent = " " * 0
base_path = "F:/GitHub/DataScience"

# Loading dataset config CSV
config_path = os.path.join(base_path, "Utils", "Configs", "datasets_config.csv")
datasets_df = pd.read_csv(config_path)

# Convert CSV config into dictionary with full paths
dataset_paths = {
    row['name']: os.path.join(base_path, row['path'].lstrip("/").lstrip("\\"))
    for _, row in datasets_df.iterrows()
}

spotify_path = dataset_paths.get('spotify_data')
spotify_dataset = pd.read_csv(spotify_path, encoding='ISO-8859-1')

# ###################################################################
# ######################### PROBLEM 1 ###############################
# ###################################################################
color_print(
    f"""
    {indent}********** PROBLEM 1. READ UNZIPPED CSV FILE. *********
    """
    , level="highlight"
)

color_print("\n Reading first few rows:\n", level="info")

spotify_zip_path = dataset_paths.get('spotify_data_zipped')
df = read_zipped_csv(spotify_zip_path)

if df is not None:
    print(df.head(10))


color_print(f"{'':48}", level="highlight")
color_print(indent + "*" * 9 + " END OF QUERYING PROBLEM 1. " + "*" * 9, level="highlight")
color_print(indent + f"{'':48}\n" * 2, level="highlight")

# ###################################################################
# ################# 2 DataSet observation ###########################
# ###################################################################
color_print(
    f"""
    {indent}********** PROBLEM 2. OBSERVATIONS AND FEATURES. *********
    {indent}********** Ensure all data has been read correctly; check the data types.********
    {indent}********** Give the columns better names. Use `apply()`. **************
    {indent}********** Get rid of any unnecessary ones. *************
    """
    , level="highlight"
)

color_print(f"\n{indent}{'*' * 9} BEGIN COLUMN MANIPULATION {'*' * 9}\n", level="highlight")
color_print(indent + "DataFrame. Initial Columns before renaming:\n", level="info")

# Ensure encoding is passed
query_dataframe(spotify_dataset, filepath=spotify_path, encoding='ISO-8859-1')

# Rename columns
spotify_dataset_original_columns = spotify_dataset.columns.copy()
spotify_dataset.columns = spotify_dataset.columns.map(snake_case)

color_print("\nOriginal column names:", level="info")
print(spotify_dataset_original_columns)

color_print("\nRenamed column names:", level="info")
print(spotify_dataset.columns)

# Convert date format
color_print(
    f"""
    {indent}Columns [release_date] can be converted to date_format='%Y-%m-%d'
    {indent}We will use custom function 'fix_dates'
    """
    , level="info"
)

spotify_dataset_date_format_changed = fix_dates(spotify_dataset, 'release_date')
print(spotify_dataset_date_format_changed.head())
print(f"{'':48}")
color_print(f"{indent}{'*' * 9} END OF QUERYING PROBLEM 2. {'*' * 9}\n\n", level="blue")
#
#
# ########################################################################
# ################# 3 Column manipulation ################################
# ########################################################################
print(
    f"""    {indent}********** PROBLEM 3. MOST PRODUCTIVE ARTIST. ***********
    {indent}********** WHO ARE THE TOP % ARTIST WITH MOST SONGS IN DATASET. ********
    {indent}********** WHOARE THE % CLEAN_MOUTH ARTISTS *******
    """
)
print(f"{'':48}")
print(indent + "*" * 9 + " BEGIN COLUMN MANIPULATION " + "*" * 9)
print(f"{'':48}\n" * 1)


color_print(
    f"""
Bellow are shown top5 artist for a different criteria
    """
    , level="info"
)

top5_artists = spotify_dataset['artist'].value_counts().head(15)
color_print(
    f"""
Top 5 artists with the most songs:
    """
    , level="info"
)
print(f"{'':48}")
print(top5_artists)


color_print(
    f"""
To print out these with most clean ( singing only for the sun, flowers, stars and mountains),
we are going to define/set clean filter which will be applied over artists
    """
    , level="info"
)

explicitly_clean_songs = spotify_dataset[spotify_dataset['explicit_track'] == 0]
text_clean_artists = explicitly_clean_songs['artist'].value_counts().head(5)
color_print(
    f"""
Top 5 clean-mouth artist artists"
    """
    , level="info"
)
print(text_clean_artists)
print(f"{'':48}\n" * 1)

print(f"{'':48}")
print(indent + "*" * 9 + " END OF QUERYING PROBLEM 3. " + "*" * 9)
print(indent + f"{'':48}\n" * 2)
#
#
# ########################################################################
# ################# 4 MOST STREAMED ARTIST ###############################
# ########################################################################
print(
    f"""    {indent}********** PROBLEM 4. MOST STREAMED ARTIST. ***********
    {indent}********** WHO ARE THE TOP 5 MOST STREAMED ARTISTS. ********
    {indent}********** ACCORDING TO SPOTIFY STREAMS *******
    """
)
print(f"{'':48}")
print(indent + "*" * 9 + " BEGIN COLUMN MANIPULATION " + "*" * 9)
print(f"{'':48}\n" * 1)


spotify_dataset = remove_commas(spotify_dataset, 'spotify_streams')
print(spotify_dataset['spotify_streams'].unique())

color_print(
    f"""
Bellow are shown top10 most streamed artist according to spotify
     """
    , level="info"
)
pd.set_option('display.float_format', '{:,.0f}'.format)

top5_most_streamed_artists = (
    spotify_dataset.groupby('artist')['spotify_streams']
    .sum()  # or use .mean(), .max(), etc. depending on your goal
    .sort_values(ascending=False)
    .head(5)
    .reset_index(name='total_streams')
)

print(top5_most_streamed_artists)

print(f"{'':48}")
print(indent + "*" * 9 + " END OF QUERYING PROBLEM 4. " + "*" * 9)
print(indent + f"{'':48}\n" * 2)

#
#
# ########################################################################
# ################# 5 SONG BY YEAR AND MONTH #############################
# ########################################################################
print(
    f"""    {indent}******* PROBLEM 5. HOW MANY SONGS PER YEAR. **********
    {indent}********** PRESENT AN APPROPRIATE PLOT. ********
    {indent}********** EXPLAIN BEHAVIOUR OF 2024 ***********
    """
)
print(f"{'':48}")
print(indent + "*" * 9 + " BEGIN COLUMN MANIPULATION " + "*" * 9)
print(f"{'':48}\n" * 1)

color_print(
    f"""
Bellow query will show how many songs are released per year
     """
    , level="info"
)

color_print(
    f"""
We have already converted 'Released date' to datetime format but lets check
     """
    , level="info"
)

print(spotify_dataset['release_date'])

color_print(
    f"""
Well not exactly, may be we forgot to overwrite the dataset, so let's do it again
     """
    , level="info"
)

spotify_dataset['release_date'] = pd.to_datetime(
    spotify_dataset['release_date'],
    errors='coerce'  # will convert invalid dates to NaT
)

print(spotify_dataset['release_date'])

color_print(
    f"""
So now that we have dates converted we can create a new column like 'release_year'
for easier work 
     """
    , level="info"
)

spotify_dataset['release_year'] = spotify_dataset['release_date'].dt.year
print(spotify_dataset['release_year'])
print(f"{'':48}\n")
color_print(
    f"""
lets count the songs per year... counting
Number of song per each year:
     """
    , level="info"
)
spotify_songs_per_year = spotify_dataset['release_year'].value_counts().sort_index()

print(spotify_songs_per_year)


plt.figure(figsize=(12, 6))
sns.barplot(x=spotify_songs_per_year.index, y=spotify_songs_per_year.values, palette='viridis')
plt.title("Number of Sons per Year")
plt.xlabel("Year")
plt.ylabel("Number of Songs")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

color_print(
    f"""
Plot shows significant drop for released songs for 2024, compared to 2023.
They are just a song shorter than year 2022.
Lets see the distribution per month
     """
    , level="info"
)

color_print(
    f"""
We will take similar if not exact same approach as we took for somgs per year.
I am not sure which representation will be more helpfull, to show month as int (1, 2, 3...)
or to use names like (jan, feb, march) and so on ...
     """
    , level="info"
)


spotify_dataset['release_month'] = spotify_dataset['release_date'].dt.month
#spotify_dataset['release_month_name'] = spotify_dataset['release_date'].dt.strftime('%B')
print(spotify_dataset['release_month'])
print(f"{'':48}\n")
color_print(
    f"""
lets count the songs per month... counting
Number of song per each month:
     """
    , level="info"
)
spotify_songs_per_month = spotify_dataset['release_month'].value_counts().sort_index()
print(spotify_songs_per_month)
print(f"{'':48}\n")
color_print(
    f"""
Now that we have songs per year and songs per months let's graph...
     """
    , level="info"
)



plt.figure(figsize=(12, 6))
sns.barplot(x=spotify_songs_per_month.index, y=spotify_songs_per_month.values, palette='viridis')
plt.title("Number of Sons per Month")
plt.xlabel("Month")
plt.ylabel("Number of Songs")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


color_print(
    f"""
We can try to plot all together songs per month per specific year
with something l ike grouped bar char
     """
    , level="info"
)


spotify_dataset['release_date'] = pd.to_datetime(
    spotify_dataset['release_date'],
    errors='coerce'
)

color_print(
    f"""
We are going to create separate columns for release_month, release_year
which we will use for grouping
     """
    , level="info"
)


spotify_dataset['release_year'] = spotify_dataset['release_date'].dt.year
spotify_dataset['release_month'] = spotify_dataset['release_date'].dt.strftime('%b')
spotify_dataset['release_month_num'] = spotify_dataset['release_date'].dt.month

grouped = spotify_dataset.groupby(['release_year', 'release_month', 'release_month_num']).size().reset_index(name='count')
print(f"{'':48}\n")
color_print(f"sorting months in calendar manner", level="info")
color_print(f"we will take values because we dont want names, rather we need ints to sort in right order", level="info")
print(f"{'':48}\n")
grouped = grouped.sort_values(['release_month_num', 'release_year'])
pivot_df = grouped.pivot(index='release_month', columns='release_year', values='count').fillna(0)
print(f"{'':48}\n")
print(pivot_df.to_string())
pivot_df.plot(figsize=(14, 7), marker='o')  # Line plot
plt.title('Number of Songs Released per Month per Year')
plt.xlabel('Month')
plt.ylabel('Number of Songs')
plt.xticks(rotation=45)
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
plt.tight_layout()
plt.show()

recent_years = list(range(2018, 2024))  # Customize as needed
pivot_df_filtered = pivot_df[recent_years]

pivot_df_filtered.plot(kind='bar', figsize=(14, 6), edgecolor='black', width=0.8)
plt.title('Number of Songs Released per Month (2018–2023)')
plt.xlabel('Month')
plt.ylabel('Number of Songs')
plt.xticks(rotation=45)
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()