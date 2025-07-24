import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import scipy.stats as st
from Utils.functions.func_fix_column_names import snake_case
from Utils.functions.func_fix_date_format import fix_dates

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.precision', 2)

Spotify_dataset = pd.read_csv("lab/Most_Streamed_Spotify_Songs_2024.csv", encoding='ISO-8859-1')

print(type(Spotify_dataset))        # should be <class 'pandas.core.frame.DataFrame'>
print('-------------------')
print(Spotify_dataset.columns)# make sure 'release_date' is here exactly
print('-------------------')
print(Spotify_dataset.head())       # see a few rows
print('-------------------')

print(Spotify_dataset)
# print(Spotify_dataset.shape)
print(Spotify_dataset.columns)
print(Spotify_dataset.dtypes)

### 2
#calling function to fix column names to snake_case
print("------------problem 3-------------")
print('------------------------------------------------')
Spotify_dataset_columns_renamed = Spotify_dataset.columns.map(snake_case)
print(Spotify_dataset_columns_renamed)
print('------------------------------------------------')
#saving data frame with new column names
Spotify_dataset.columns = Spotify_dataset.columns.map(snake_case)
print(Spotify_dataset)
print('------------------------------------------------')
print(type(Spotify_dataset))
print(Spotify_dataset.columns)

#calling function to fix date format
Spotify_dataset_date_format_changed = fix_dates(Spotify_dataset, 'release_date')
print(Spotify_dataset_date_format_changed.head())


###  Get rid of any unnecessary ones.
### based  on the other requst and observing the columns i thing the only columns that can be dropped as unneceserry ar

### 3
print("------------problem 3-------------")
Spotify_dataset_Explicitlely_clean = Spotify_dataset[Spotify_dataset['explicit_track'] == 0]
print(Spotify_dataset_Explicitlely_clean)

Spotify_dataset_artist_song_amount = Spotify_dataset_Explicitlely_clean.groupby('artist')['track'].count()
print(Spotify_dataset_artist_song_amount)
### problem encautered when this printed out the artist. Some ygly encoding showed up
''''

ýýýýýýýýýýýýýýý ýýýýýýýýýýýýýýý                1
ýýýýýýýýýýýýýýý(ýýýýýýýýýýýýýýý)               1
ýýýýýýýýýýýýýýýýýý                             1
ýýýýýýýýýýýýýýýýýýýýý ýýýýýýýýýýýýýýýýýýýýý    1
ýýýýýýýýýýýýýýýýýýýýýýýý  

'''

# steps taken -> changed encoding durign the pandas reading csv to 'utf-8'
## i got another error: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xfd in position 2679: invalid start byte
### changed encoding to -> encoding='ISO-8859-1' -> same ugly ugliness

''''

ýýýýýýýýýýýýýýý ýýýýýýýýýýýýýýý                1
ýýýýýýýýýýýýýýý(ýýýýýýýýýýýýýýý)               1
ýýýýýýýýýýýýýýýýýý                             1
ýýýýýýýýýýýýýýýýýýýýý ýýýýýýýýýýýýýýýýýýýýý    1
ýýýýýýýýýýýýýýýýýýýýýýýý  

'''

