import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re


coffee_data = pd.read_csv("lab/merged_data_cleaned.csv")
print(coffee_data)


def to_kg(value):
    if pd.isna(value):
        return None

    # convert to string and clean
    unitString = str(value).lower().strip().replace(" ", "")

    if "kg,lbs" in unitString:
        matchValue = re.match(r"([0-9\.]+)", unitString)
        if matchValue:
            return round(float(matchValue.group(1)))
        else:
            return None

    if unitString in ["0", "0kg", "0lbs"]:
        return 0

    matchValue = re.match(r"([0-9\.]+)", unitString)
    if not matchValue:
        return None

    num = float(matchValue.group(1))

    if "kg" in unitString:
        return round(num)
    elif "lbs" in unitString:
        return round(num * 0.453592)
    else:
        return round(num)


coffee_data["Bag.Weight [kg]"] = coffee_data["Bag.Weight"].apply(to_kg)
print(coffee_data[["Bag.Weight", "Bag.Weight [kg]"]])#.head(10))

##### Plotting Data #####
q1 = coffee_data["Bag.Weight [kg]"].quantile(0.25)
q3 = coffee_data["Bag.Weight [kg]"].quantile(0.75)
iqr = q3 - q1

#boudries
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 - 1.5 * iqr

# flag outliars
coffee_data["is_outliar"] = ~coffee_data["Bag.Weight [kg]"].between(lower_bound, upper_bound)

plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=coffee_data,
    x=coffee_data.index,
    y="Bag.Weight [kg]",
    hue="is_outliar",
    palette={True: "red", False: "green"},
    alpha=0.7
)

plt.title("Bag Weight [kg] with Outliers Highlighted")
plt.xlabel("Index")
plt.ylabel("Weight (kg)")
plt.legend(title="Outlier")
plt.grid(True)
plt.tight_layout()
plt.show()

