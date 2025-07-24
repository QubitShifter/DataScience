import numpy as np
import pandas as pd

Spotify_dataset = pd.read_csv("lab/Most_Streamed_Spotify_Songs_2024.csv", encoding='cp1252')


def snake_case(column):
    return(
        column.replace(' ', '_')
              .replace('-', '_')
              .replace('.', '_')
              .lower()
    )


Spotify_dataset_columns_renamed = Spotify_dataset.columns.map(snake_case)
print(Spotify_dataset_columns_renamed)
