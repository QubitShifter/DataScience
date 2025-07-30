from Utils.Functions.func_helper_print_colors import color_print
from Utils.Functions.func_helper_duplicate_rows import print_duplicate_rows
import chardet
import pandas as pd
import os


def query_dataframe(dataframe, filepath=None, encoding=None, indent="    "):
    # Encoding + filepath diagnostics
    if filepath:
        if encoding:
            color_print(f"{indent}Encoding check:", level="info")
            color_print(f"{indent} Filepath: {filepath}", level="info")
            color_print(f"{indent} Encoding: {encoding}", level="info")
        else:
            color_print(f"{indent} Filepath provided but no encoding specified.", level="yellow")

        # Skip raw reading if ZIP
        if filepath.lower().endswith('.zip'):
            color_print(f"{indent} Skipping raw line count: ZIP file detected.", level="info")
        else:
            try:
                with open(filepath, 'r', encoding=encoding or 'utf-8') as f:
                    total_lines = sum(1 for _ in f)
                color_print(f"{indent} Total lines in file (including header): {total_lines}", level="info")
                color_print(f"{indent} Rows loaded in DataFrame: {dataframe.shape[0]}", level="info")
                if dataframe.shape[0] < total_lines - 1:
                    color_print(f"{indent} Warning: Some rows may not have been read.", level="yellow")
            except Exception as e:
                color_print(f"{indent} Could not read raw file to count lines: {e}", level="yellow")
    else:
        color_print(f"{indent}No filepath provided.", level="yellow")

    color_print(f"\n{indent}Dataset Summary\n{indent}{'-' * 50}", level="info")

    # Basic info
    color_print(f"{indent} DataFrame dimensions: {dataframe.shape}", level="info")
    print(f"{'':48}")

    # Null values
    nulls = dataframe.isnull().sum()
    if nulls.sum() > 0:
        color_print(f"{indent} Missing values per column:", level="warning")
        print(f"{'':48}")
        print(nulls[nulls > 0])
    else:
        color_print(f"{indent} No missing values detected.", level="info")
    print(f"{'':48}")

    # Duplicates
    print_duplicate_rows(dataframe, indent=indent)
    print(f"{'':48}")

    # Columns and dtypes
    color_print(f"\n{indent}Columns and data types:", level="info")
    for col, dtype in dataframe.dtypes.items():
        print(f"{indent}  - {col}: {dtype}")

    # Column listing
    color_print(f"\n{indent}Column names (grouped):", level="info")
    cols = list(dataframe.columns)
    for i in range(0, len(cols), 5):
        print(indent + "  " + str(cols[i:i + 5]))

    # Preview rows
    color_print(f"\n{indent}First 5 rows:", level="info")
    print(dataframe.head())

    color_print(f"\n{indent}Last 5 rows:", level="info")
    print(dataframe.tail())

    # Summary stats
    color_print(f"\n{indent}Descriptive statistics:", level="info")
    print(dataframe.describe(include='all', datetime_is_numeric=True))
