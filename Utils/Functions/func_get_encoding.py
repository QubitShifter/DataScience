import chardet
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.precision', 2)


def encoding_pre_check(filepath, read_rows=False):
    try:
        #with open("F:/GitHub/Python/DataScience/05.DataVisualisation/lab/Most_Streamed_Spotify_Songs_2024.csv", 'rb') as f:
        with open(filepath, 'rb') as f:
            raw_data = f.read(100000)
            result = chardet.detect(raw_data)

        encoding = result['encoding']
        confidence = result['confidence']

        print(f"Used encoding for loaded dataset is: {encoding} (confidence: {confidence:.2f})")


        try:
            dataframe = pd.read_csv(filepath, encoding=encoding)

            print("CSV file is loaded successfully with detected encoding.")
            if read_rows:
                print("\n Showing bunch of rows:")
                print(dataframe.head(5))

        except UnicodeDecodeError as ude:
            print(f" Encoding Errors while loading file: {ude}")

        except Exception as e:
            print(f" Different errors while reading CSV: {e}")

    except Exception as e:
        print(f" Failed to open and read file: {e}")
