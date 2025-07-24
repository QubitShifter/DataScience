import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os

coffee_data = pd.read_csv("lab/merged_data_cleaned.csv")
print(coffee_data)


def snake_case(column):
    return(
        column.replace('-', '_')
              .replace(' ', '_')
              .replace('.', '_')
              .lower()
    )


coffee_data.columns = [snake_case(column) for column in coffee_data.columns]
print(coffee_data.columns)