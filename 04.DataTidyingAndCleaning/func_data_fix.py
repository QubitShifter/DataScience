import numpy as np
import pandas as pd
import re

# Load the dataset
coffee_data = pd.read_csv("lab/merged_data_cleaned.csv")

#
# # --- Mappings ---
# month_map = {
#     'january': '01', 'february': '02', 'march': '03', 'april': '04',
#     'may': '05', 'june': '06', 'july': '07', 'august': '08',
#     'september': '09', 'sept': '09', 'october': '10',
#     'november': '11', 'december': '12',
#     'enero': '01', 'febrero': '02', 'marzo': '03', 'abril': '04',
#     'mayo': '05', 'junio': '06', 'julio': '07', 'agosto': '08',
#     'septiembre': '09', 'octubre': '10', 'noviembre': '11', 'diciembre': '12'
# }
#
# quarter_map = {
#     '1t': '01', 'q1': '01',
#     '2t': '04', 'q2': '04',
#     '3t': '07', 'q3': '07',
#     '4t': '10', 'q4': '10',
# }
#
#
# # --- Functions ---
# def convert_quarter_format(value):
#     if isinstance(value, str) and "T" in value:
#         parts = value.split("/")
#         quarter = parts[0][0] if parts else None
#         try:
#             year = parts[1]
#             if len(year) == 2:
#                 year = f"20{year}"
#             return f"Q{quarter} {year}"
#         except:
#             return np.nan
#     return value
#
#
# def convert_to_mm_yyyy(value):
#     if pd.isna(value):
#         return None
#     value = str(value).lower().strip()
#
#     if re.fullmatch(r"\d{4}", value):
#         return f"01-{value}"
#
#     match = re.match(r"(\d{4})\s*[-/]\s*(\d{4})", value)
#     if match:
#         return f"01-{match.group(1)}"
#
#     match = re.match(r"(1t|2t|3t|4t|q1|q2|q3|q4)[/ ]*(\d{2,4})", value)
#     if match:
#         quarter = quarter_map.get(match.group(1))
#         year = match.group(2)
#         if len(year) == 2:
#             year = f"20{year}"
#         return f"{quarter}-{year}"
#
#     for month_name, mm in month_map.items():
#         if month_name in value:
#             match = re.search(r"(\d{4})", value)
#             if match:
#                 return f"{mm}-{match.group(1)}"
#
#     match = re.search(r"(\d{4})", value)
#     if match:
#         return f"01-{match.group(1)}"
#
#     return None
#
# def convert_quarter_format_and_quarter(value):
#     if pd.isna(value):
#         return pd.Series([None, None])
#
#     s = str(value).strip().lower()
#
#     match = re.match(r"([1-4]t|q[1-4])[/\-]?(\d{2,4})", s)
#     if match:
#         quarter_key = match.group(1)
#         year = match.group(2)
#         if len(year) == 2:
#             year = '20' + year
#
#         month = quarter_map.get(quarter_key)
#         if month:
#             formatted_date = f"{month}-{year}"
#             quarter_label = f"Q{((int(month) - 1) // 3 + 1)}"
#             return pd.Series([formatted_date, quarter_label])
#
#     return pd.Series([None, None])
#
#
# # --- Apply all transformations ---
# coffee_data["Harvest.QuarterLabel"] = coffee_data["Harvest.Year"].apply(convert_quarter_format)
# coffee_data["Harvest.MMYYYY"] = coffee_data["Harvest.Year"].apply(convert_to_mm_yyyy)
# coffee_data[["Harvest.MMYYYY_fromQuarter", "Harvest.Quarter"]] = coffee_data["Harvest.Year"].apply(convert_quarter_format_and_quarter)
#
# # --- Print actual data preview ---
# print(coffee_data[[
#     "Harvest.Year",
#     "Harvest.QuarterLabel",
#     "Harvest.MMYYYY",
#     "Harvest.MMYYYY_fromQuarter",
#     "Harvest.Quarter"
# ]])

print(coffee_data.shape)
print(coffee_data.columns)
#print(coffee_data['Unnamed: 0'].is_unique)
#print(coffee_data['Unnamed: 0'].value_counts())
#print(coffee_data["Unnamed: 0"].unique())
#print(coffee_data["Harvest.Year"].unique())
print(coffee_data["Harvest.Year"].value_counts())
print(coffee_data[coffee_data["Harvest.Year"].str.contains(r'\d{4}', na=False)]["Harvest.Year"])

#print(coffee_data["Harvest.Year"].str.contains(r'\d{4}', na=False))


print("----------------------------------")

# --- Mappings ---
month_map = {
    'january': '01', 'february': '02', 'march': '03', 'april': '04',
    'may': '05', 'june': '06', 'july': '07', 'august': '08',
    'september': '09', 'sept': '09', 'october': '10',
    'november': '11', 'december': '12'
}

quarter_map = {
    '1t': '01', 'q1': '01',
    '2t': '04', 'q2': '04',
    '3t': '07', 'q3': '07',
    '4t': '10', 'q4': '10',
}


def split_HarvertColumn_into_Two(value):
    if pd.isna(value):
        return pd.Series([None, None])

    value = str(value).strip().lower()

    # year_match = re.search(r'\b(\d{4})\b', value)
    # year = year_match.group(1) if year_match else None

    ## Year
    if re.fullmatch(r"\d{4}", value):
        return f"01-{value}"

    ## Year cpmbination
    year_match = re.match(r"d{4}\s*[-/]\s(\d{4})]", value)
    if year_match:
        return f"01-{year_match.group(0)}"


    quarter_match = re.search(r'(q[1-4]|[1-4]t)', value)
    if quarter_match:
        period = quarter_match.group(1).upper().replace('T', '')
    else:
        period_match = re.search(
            r'(spring|summer|fall|autumn|winter|january|february|march|'
            r'april|may|june|july|august|september|october|november|december)',
            value
        )
        period = period_match.group(1).capitalize() if period_match else None

    return pd.Series([year, period])


coffee_data_CleanYear = coffee_data["Harvest.Year"].apply(split_HarvertColumn_into_Two)
print(coffee_data_CleanYear[0].unique())
print(coffee_data_CleanYear[1].unique())

















# print("----------------------------------")
#
# def convert_to_mm_yyyy(value):
#     if pd.isna(value):
#         return None
#
#     value = str(value).lower().strip()
#
# #get only year
#
#     if re.fullmatch(r"\d{4}", value):
#         return f"01-{value}"
#
# coffee_data["Harvest.Year_changed"] = coffee_data["Harvest.Year"].apply(convert_to_mm_yyyy)
# print(coffee_data["Harvest.Year_changed"])