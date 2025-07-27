import chardet

def query_dataframe(dataframe, filepath=None, indent="    "):
    print(f"{'':48}")
    print(indent + "********* PROBLEM 1. READ THE DATASET **********")
    print(indent + "********* Begin dataset observation ************")
    print(f"{'':48}\n")

    if filepath:
        with open(filepath, 'rb') as f:
            result = chardet.detect(f.read(100000))
        print(f"Pre-check. Detected CSV encoding: {result}")
    else:
        print("No filepath provided for encoding check.")

    print(
        f"""
            Pre-check. get csv encoding
        """
    )

    with open("F:/GitHub/Python/DataScience/05.DataVisualisation/lab/Most_Streamed_Spotify_Songs_2024.csv", 'rb') as f:
        result = chardet.detect(f.read(100000))
        print(result)

    print(
        f"""
            DataSet encodinng is: {result}
        """
    )


    # Basic info
    print(indent + f"DataFrame.dimensions: {dataframe.shape}")
    print(f"{'':48}")
    print(indent + "DataFrame.coluns and rows:")
    print(f"{'':48}")
    print(dataframe)
    print(f"{'':48}")
    print(dataframe.columns)
    print(f"{'':48}")
    print(dataframe.dtypes)
    print(f"{'':48}\n" * 2)

    # Column listing
    print(indent + "DataFrame's Initial Columns:\n")
    print(f"--------------------------------------")
    cols = list(dataframe.columns)
    for i in range(0, len(cols), 5):
        print(cols[i:i+5])
    print(f"--------------------------------------")
    print(f"{'':48}")
    print(f"--------------------------------------")

    # Data types again
    print(indent + "\nData Types:\n", dataframe.dtypes)
    print(indent + f"--------------------------------------")
    print(f"{'':48}")
    print(indent + f"--------------------------------------")

    # First rows
    print("\nFirst few rows:\n", dataframe.head())
    print(f"{'':48}")

    # Summary statistics
    print(indent + "\nDescriptive statistics (numerical only):")
    print(f"{'':48}")
    print(dataframe.describe())

    print(f"{'':48}")
    print(indent + "*" * 9 + " END OF QUERYING PROBLEM 1. " + "*" * 9)
    print(indent + f"{'':48}\n" * 2)
