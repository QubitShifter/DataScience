import chardet

with open("F:/GitHub/Python/DataScience/05.DataVisualisation/lab/Most_Streamed_Spotify_Songs_2024.csv", 'rb') as f:
    result = chardet.detect(f.read(100000))
    print(result)
