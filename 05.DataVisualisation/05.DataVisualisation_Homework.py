import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import scipy.stats as st
from Utils.Functions.func_helper_query_dataset import query_dataframe
from Utils.Functions.func_fix_column_names import snake_case
from Utils.Functions.func_fix_date_format import fix_dates
from Utils.Functions.func_helper_remove_commas import remove_commas
from Utils.Functions.func_get_encoding import encoding_pre_check
from Utils.Functions.func_helper_query_dataset import query_dataframe


indent = " " * 0
base_path = "F:/GitHub/DataScience"

config_path = os.path.join(base_path, "Utils", "Configs", "datasets_config.csv")
datasets_df = pd.read_csv(config_path)
dataset_paths = dict(zip(datasets_df['name'], datasets_df['path']))

#load spotify_dataset
spotify_path = dataset_paths['spotify_data']
spotify_dataset = pd.read_csv(spotify_path, encoding='ISO-8859-1')

###################################################################
###################### 1 Reading Data #############################
###################################################################

print(query_dataframe(spotify_dataset))
print(query_dataframe(spotify_dataset, filepath=spotify_path))


###################################################################
################# 2 DataSet observation ###########################
###################################################################
print(
    f"""
    {indent}********** PROBLEM 2. OBSERVATIONS AND FEATURES. ********* 
    {indent}********** Ensure all data has been read correctly; check the data types.********
    {indent}********** Give the columns better names. Use `apply()`. **************
    {indent}********** Get rid of any unnecessary ones. *************
    """
)

print(f"{'':48}")
print(indent + "*" * 9 + " BEGIN COLUMN MANIPULATION " + "*" * 9)
print(f"{'':48}\n" * 1)
print(indent + "DataFrame. Initial Columns before renaming:\n")

spotify_dataset_column_rename = spotify_dataset.columns.map(snake_case)
spotify_dataset_column_original_column_names = spotify_dataset.columns.copy()
spotify_dataset.columns = spotify_dataset_column_rename

print(spotify_dataset_column_original_column_names)
print(f"{'':48}\n" * 1)
print("DataFrame. Columns after being renamed:\n")
print(spotify_dataset.columns)

print(
    f"""
        Columns [release_date] can be converted to date_format='%Y-%m-%d'
        We will use custom function 'fix_dates'
    """
)

spotify_dataset_date_format_changed = fix_dates(spotify_dataset, 'release_date')
print(spotify_dataset_date_format_changed.head())

print(f"{'':48}")
print(indent + "*" * 9 + " END OF QUERYING PROBLEM 2. " + "*" * 9)
print(indent + f"{'':48}\n" * 2)


########################################################################
################# 3 Column manipulation ################################
########################################################################
print(
    f"""    {indent}********** PROBLEM 3. MOST PRODUCTIVE ARTIST. *********** 
    {indent}********** WHO ARE THE TOP % ARTIST WITH MOST SONGS IN DATASET. ********
    {indent}********** WHOARE THE % CLEAN_MOUTH ARTISTS ******* 
    """
)
print(f"{'':48}")
print(indent + "*" * 9 + " BEGIN COLUMN MANIPULATION " + "*" * 9)
print(f"{'':48}\n" * 1)


print(
    f"""
        Bellow are shown top5 artist with the most songs in the dataset
    """
)

top5_artists = spotify_dataset['artist'].value_counts().head(15)
print("Top 5 artists with the most songs:")
print(top5_artists)


print(
    f"""
        To print out these with most clean ( singing only for the sun, flowers, stars and mountains),
        we are going to define/set clean filter which will be applied over artists
    """
)

explicitly_clean_songs = spotify_dataset[spotify_dataset['explicit_track'] == 0]
text_clean_artists = explicitly_clean_songs['artist'].value_counts().head(5)
print(f"\nTop 5 clean-mouth artist artists")
print(text_clean_artists)
print(f"{'':48}\n" * 1)

print(f"{'':48}")
print(indent + "*" * 9 + " END OF QUERYING PROBLEM 3. " + "*" * 9)
print(indent + f"{'':48}\n" * 2)


########################################################################
################# 4 Column manipulation ################################
########################################################################
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

print(
    f"""
        Bellow are shown top10 most streamed artist according to spotify
    """
)
pd.set_option('display.float_format', '{:,.0f}'.format)

top5_most_streamed_artists = (
    spotify_dataset.groupby('artist')['spotify_streams']
    .sum()  # or use .mean(), .max(), etc. depending on your goal
    .sort_values(ascending=False)
    .head(15)
)

print(top5_most_streamed_artists)





# spotify_dataset_artist_song_amount = spotify_dataset_Explicitlely_clean.groupby('artist')['track'].count()
# print(spotify_dataset_artist_song_amount)
### problem encautered when this printed out the artist. Some ygly encoding showed up
# ''''
#
# ýýýýýýýýýýýýýýý ýýýýýýýýýýýýýýý                1
# ýýýýýýýýýýýýýýý(ýýýýýýýýýýýýýýý)               1
# ýýýýýýýýýýýýýýýýýý                             1
# ýýýýýýýýýýýýýýýýýýýýý ýýýýýýýýýýýýýýýýýýýýý    1
# ýýýýýýýýýýýýýýýýýýýýýýýý
#
# '''
#
# # steps taken -> changed encoding durign the pandas reading csv to 'utf-8'
# ## i got another error: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xfd in position 2679: invalid start byte
# ### changed encoding to -> encoding='ISO-8859-1' -> same ugly ugliness
#
# ''''
#
# ýýýýýýýýýýýýýýý ýýýýýýýýýýýýýýý                1
# ýýýýýýýýýýýýýýý(ýýýýýýýýýýýýýýý)               1
# ýýýýýýýýýýýýýýýýýý                             1
# ýýýýýýýýýýýýýýýýýýýýý ýýýýýýýýýýýýýýýýýýýýý    1
# ýýýýýýýýýýýýýýýýýýýýýýýý
#
# '''

