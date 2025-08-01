import re
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
from Utils.Functions.func_plot_graphs import plot_graphs
from Utils.Functions.func_helper_print_colors import color_print
from Utils.Functions.func_helper_query_dataset import query_dataframe
from Utils.Functions.func_fix_column_names import snake_case
from Utils.Functions.func_fix_date_format import fix_dates
from Utils.Functions.func_helper_remove_commas import remove_commas
from Utils.Functions.func_helper_query_dataset import query_dataframe
from Utils.Functions.func_read_csv_zipped import read_zipped_csv
from Utils.Functions.func_helper_clean_numeric_columns import clean_numeric_columns
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

print(f"{'':48}")
print(indent + "*" * 9 + " END OF QUERYING PROBLEM 5. " + "*" * 9)
print(indent + f"{'':48}\n" * 2)

#
#
# ########################################################################
# ############################ 6 PLAYLISTS ###############################
# ########################################################################
print(
    f"""    {indent}*********** PROBLEM 6. PLAYLISTS. ***************
    {indent}*** CORRELATIONS BETWEEN ADDING A SONGS TO PLAYLISTS. ***
    {indent}************** PLOT ALL RELATIONSHIPS *******************
    """
)
print(f"{'':48}")
print(indent + "*" * 9 + " BEGIN COLUMN MANIPULATION " + "*" * 9)
print(f"{'':48}\n" * 1)

color_print(
    f"""
For the purpose pf this task we will create a playlist dict containing
related columns that we are going to use
     """
    , level="info"
)

playlist_columns = [
    'spotify_playlist_count',
    'apple_music_playlist_count',
    'deezer_playlist_count',
    'amazon_playlist_count'
]

playlist_df = spotify_dataset[playlist_columns].copy()
print(playlist_df)

color_print(
    f"""
few problems exists after looking at the 'playlist_df' values.
Values like '269,802' are comma separated, they will be interpreted as a
string. ALsi there are NaNs in the columns. In general the 'table' in this 
format cannot be used for finding correlations or plotting. 
     """
    , level="info"
)

color_print(
    f"""
So before starting whatever we want to start numbers should be converted to ints,
NaNs should be dropped or handled in some other way. 
     """
    , level="info"
)
print(f"{'':48}")
color_print(f"lets see original values", level="info")
for cols in playlist_columns:
    color_print(f"\n---{cols} (before handling the data)", level="info")
    print(spotify_dataset[cols].head(10))

playlist_df = spotify_dataset[playlist_columns].copy()
pd.reset_option('display.float_format')
for cols in playlist_columns:
    playlist_df[cols] = (
        playlist_df[cols]
        .astype(str)
        .str.replace(',', '', regex=False)
        .replace(['nan', 'NaN'], pd.NA)  # Handle both string cases
    )
    playlist_df[cols] = pd.to_numeric(playlist_df[cols], errors='coerce')

    color_print(f"\n---{cols} (after handling the data)", level="info")
    print(playlist_df['spotify_playlist_count'].head(10).to_list())
    print(playlist_df[cols].head(10))



color_print(
    f"""
The question now is: how to handle NaNs / NAs... In order to be able to 
visualize or make correlations, we need clean data. Plots and correlations
dont like NaNs / NAs...   
     """
    , level="info"
)

color_print(
    f"""
One alternative way i to replace NaNs with ZERO. Thih is assumption that the 
song has zero (0) playlists. 
We can also try Pearson correlation which will handle the NaN
     """
    , level="info"
)

color_print(
    f"""
Lets try Pearson with NaNs
     """
    , level="info"
)

NaN_correlation = playlist_df.corr(method='pearson')

plt.figure(figsize=(8, 5))
sns.heatmap(NaN_correlation, annot=True, cmap='coolwarm', fmt=".2f", vmin=0, vmax=1)
plt.title("Correlation Matrix (NaN Ignored)")
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()


color_print(
    f"""
So, how can we interpretate the result in this correlation map..
apple_music <-> deezer - Corelation 0.78 -> strong positive relationship: When a particular song is in many Apple playlists, it is often also in many Deezer playlists.
spotify <-> apple_music - Correlation 0.69 -> moderately strong: May be popular songs on Spotify playlists are also  on Apple Music.
deezer<-> amazon - Correlation 0.57 -> moderate relationship: there is some overlap, but it is rather weak
spotify <-> amazon - Correlation 0.38 -> weak correlation

     """
    , level="info"
)

color_print(
    f"""
This can mean that correlation in not equal to causation.. we cannot say loudly that one platform affects another directly.
Bit a point to make, this is only linear correlation, there might be a differences in correlation with some other methods.
also we are dealing with NaNs. 
NaN handling matters — we might get slightly different correlations if we replace NaNs with 0s.
     """
    , level="info"
)
color_print(
    f"""
Lets test NaNs repalced with Zeroes
     """
    , level="info"
)

playlist_df_zero = playlist_df.fillna(0)
correlation_matrix_zeroes = playlist_df_zero.corr(method='pearson')

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix_zeroes, annot=True, cmap='coolwarm', fmt=".2f", vmin=0, vmax=1)
plt.suptitle("Correlation Matrix (NaNs Replaced with Zeroes)")
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()


color_print(
    f"""
Lets try to compare both methods and find the differences
     """
    , level="info"
)

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
sns.heatmap(NaN_correlation, ax=axes[0], annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
axes[0].set_title("Correlation (with NaNs)")

sns.heatmap(correlation_matrix_zeroes, ax=axes[1], annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
axes[1].set_title("Correlation (NaNs replaced with 0)")

plt.tight_layout()
plt.show()


difference_matrix = correlation_matrix_zeroes - NaN_correlation
difference = NaN_correlation.copy()

for col in NaN_correlation.columns:
    for row in NaN_correlation.index:
        val_nan = NaN_correlation.loc[row, col]
        val_zero = correlation_matrix_zeroes.loc[row, col]
        diff = val_zero - val_nan
        difference.loc[row, col] = f"{val_nan}:.2f -> {val_zero:.2f} (diff {diff:+.2f})"

color_print("\n Correlation Comparison between bot methods (NaNs vs Zeroes):\n", level="highlight")
print(difference)


color_print(
    f"""
We can try to find % of missing values per column
     """
    , level="highlight"
)
print(f"{'':48}")
persent_of_missing_values = playlist_df.isna().mean() * 100
color_print(
    f"""
Percentage of missing values per column:
     """
    , level="highlight"
)
print(f"{'':48}")
print(persent_of_missing_values)
color_print(
    f"""
What we see is that Spotify has almost no missing data.
Apple has small amount of missing data, but still can make a difference in some stats.
Amazon and Deezer are between 20 and 25 % which is more significant 
     """
    , level="highlight"
)

color_print(
    f"""
So, replacing NaNs with 0 here is introducing some artificial inflation of correlation, 
especially with Spotify or Apple data that has low missingness.

So if we want stats accuracy it is better to stick with NaNs
     """
    , level="warning"
)

print(f"{'':48}")
print(indent + "*" * 9 + " END OF QUERYING PROBLEM 6. " + "*" * 9)
print(indent + f"{'':48}\n" * 2)

#
#
# ########################################################################
# ############################# 7 YOUTUBE ################################
# ########################################################################
print(
    f"""    {indent}****** PROBLEM 7. YOUTUBE VIEWS AND LIKES. ***********
    {indent}******** RELATIONSHIP BETWEEN YOUTUBE VIRES AND LIKES. *******
    {indent}****** WHAT IS THE MEAN YOUTUBE VIEWS-TO-LIKE RATIO **********
    """
)
print(f"{'':48}")
print(indent + "*" * 9 + " BEGIN COLUMN MANIPULATION " + "*" * 9)
print(f"{'':48}\n" * 1)

youtube_playlist_columns = [
    'youtube_views',
    'youtube_likes',
    'youtube_playlist_reach'
]

youtube_playlist_df = spotify_dataset[youtube_playlist_columns].copy()
print(youtube_playlist_df)

color_print(
    f"""
Again we will need to clean commas and mace columns numerical. 
Well.. will make a func to convert commas columns to numeric columns since this is second time 
that i need to do the same thing just for different columns. -> func: clean_numeric_columns
     """
    , level="info"
)

youtube_playlist_numeric_columns = clean_numeric_columns(youtube_playlist_df, youtube_playlist_columns)
print(youtube_playlist_numeric_columns)

color_print(
    f"""
Now that we have numeric values in our columns, lets plot
We can use scatter plot to show spread of data points for views and likes.
This will help with better visualisation
     """
    , level="info"
)


color_print(
    f"""
So this is N-tieth time to write the same plt.whatever plot we want.. 
This means its better to have a plotting function...
     """
    , level="info"
)


plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")

# Scatter plot with regression line
sns.regplot(
    data=youtube_playlist_numeric_columns,
    x="youtube_views",
    y="youtube_likes",
    scatter_kws={'alpha': 0.5},
    line_kws={'color': 'red'}
)

# Optional: Log scale for better spread if needed
plt.xscale('log')
plt.yscale('log')

# Labels and title
plt.xlabel("YouTube Views (log scale)")
plt.ylabel("YouTube Likes (log scale)")
plt.title("Relationship Between YouTube Views and Likes")

plt.tight_layout()
plt.show()


plot_graphs(
    df=youtube_playlist_numeric_columns,
    x_col='youtube_views',
    y_col='youtube_likes',
    plot_type='scatter',
    regression=True,
    title='Youtube Views vs Likes Regression'
)


plot_graphs(
    df=youtube_playlist_numeric_columns,
    x_col='youtube_views',
    y_col='youtube_likes',
    plot_type='scatter',
    regression=True,
    title='YouTube Views vs Likes with Correlation',
    xlabel='YouTube Views',
    ylabel='YouTube Likes',
    log_scale=True,
    show_corr=True,
    remove_outliers=True,
    outlier_method='iqr',
    save_path='views_vs_likes_plot.png'
)


color_print(
    f"""
What we can say while staring at the graph. We have regression line that estimates the overall trend.
We can say that as views increase, likes generally increase too — we say that we have positive correlation.
The line in curved which means that probably likes increase non linearly with views.
     """
    , level="info"
)

color_print(
    f"""
We can also notice that there are tracks with huge views but relatively low likes, 
which pull the regression line slightly down.
There are also a few very low-liked tracks with high views might be anomalies, or just a really BAD songs
     """
    , level="info"
)

youtube_ratio_df = youtube_playlist_numeric_columns.copy()
youtube_ratio_df = youtube_ratio_df[youtube_ratio_df['youtube_likes'] > 0]

youtube_ratio_df['views_to_likes_ratio'] = youtube_ratio_df['youtube_views'] / youtube_ratio_df['youtube_likes']

youtube_mean_ration = youtube_ratio_df['views_to_likes_ratio'].mean()
youtube_median_ration = youtube_ratio_df['views_to_likes_ratio'].median()

color_print(f"Youtube mean ratio: {youtube_mean_ration:,.2f}", level="info")
color_print(f"Youtube median ratio: {youtube_median_ration:,.2f}", level="info")


plot_graphs(
    df=youtube_ratio_df,
    plot_type = 'boxplot',
    x_col='views_to_likes_ratio',
    y_col=None,
    title='Distribution of YouTube Views-to-Likes Ratio',
    ylabel='Views per Like',
    remove_outliers=False
)

plot_graphs(
    df=youtube_ratio_df,
    plot_type = 'boxplot',
    x_col=None,
    y_col='views_to_likes_ratio',
    title='Distribution of YouTube Views-to-Likes Ratio',
    ylabel='Views per Like',
    log_scale=True,
    remove_outliers=False
)


plot_graphs(
    df=youtube_ratio_df,
    plot_type='hist',
    x_col='views_to_likes_ratio',
    title='Distribution of YouTube Views-to-Likes Ratio',
    xlabel="Views per Like",
    ylabel='Frequency"',
    figsize=(10, 6)
)

color_print(
    f"""
това което виждаме на графиката, е един пик в ляво, при това доста голям, което може да каже, че по-голямата част от стойностите са
доста ниски ( много видеа имат доста ниска отношение 'view-to-likes'.
Това може и да е проблем на начина по-който плотваме данните.
     """
    , level="info"
)

#chi-square