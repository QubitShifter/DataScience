import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import scipy.stats as st




ai_dataset = pd.read_csv("lab/ai_job_dataset.csv")

"""""
print(ai_dataset.columns)
print("----------------")
print(ai_dataset.duplicated("job_title"))
print("----------------")

print(ai_dataset.shape)
print("----------------")
print(ai_dataset.dtypes)
print("----------------")
print(ai_dataset.sample(5))
print("----------------")
print(ai_dataset.job_id.nunique()) # дали има повтарящи се
print(ai_dataset.job_id.nunique() == len(ai_dataset)) # броя на уникалните елементи е равен на дължината на ai_dataset
print("----------------")
print(ai_dataset.index)
ai_dataset.set_index("job_id")
print(ai_dataset.index)
print("----------------")



print(ai_dataset[
    ai_dataset.job_id == "AI05713"
])
print("----------------")

print(ai_dataset.salary_currency.value_counts(dropna= False))
print("----------------")


print(ai_dataset["company_location"].value_counts())
print(ai_dataset["remote_ratio"].value_counts())
print("----------------")

print(ai_dataset["remote_ratio"].replace({0: "Onsite", 50: "Hybrid", 100: "Fully Remote"}))
print("----------------")

print(ai_dataset["experience_level"].astype("category"))
print("----------------")

ai_dataset["experience_level"] = ai_dataset["experience_level"].astype("category")
print("----------------")

print(ai_dataset["experience_level"])
print(ai_dataset["experience_level"].value_counts())

ai_dataset_pivot = ai_dataset.pivot_table(columns= "experience_level", index = "company_size", aggfunc= "size")
print(ai_dataset_pivot)

print(st.chi2_contingency(ai_dataset.pivot_table(columns= "experience_level", index = "company_size", aggfunc= "size")))
print("----------------")
print("----------------")
print(ai_dataset["job_title"])

print(ai_dataset["job_title"].value_counts())

"""
print(ai_dataset["company_location"])
print("----------------------------")

print(ai_dataset["salary_currency"].value_counts(dropna=False))
print("------------------------------------------------------")

print(ai_dataset["company_location"].value_counts())
print("-------------------------------------------")


print(ai_dataset.columns)
print(ai_dataset["remote_ratio"])
print("-------------------------------------------")
print(ai_dataset["remote_ratio"].replace({0: "Onsite", 50: "Hybrid", 100: "Full"}))

print(ai_dataset.experience_level.astype("category"))

ai_dataset.experience_level = ai_dataset.experience_level.astype("category")
print("-------------------------------------------------------------------")
print("-------------------------------------------------------------------")
print(ai_dataset.experience_level.value_counts())

print("--------------------------Pivot table---------------------------------")
print(ai_dataset.pivot_table(columns="experience_level", index="company_size", aggfunc = "size"))

print(st.chi2_contingency(ai_dataset.pivot_table(columns="experience_level", index="company_size", aggfunc = "size")))
print("-------------------------------------------------------------------------------------------------------------")
print(ai_dataset.job_title)
print("-------------------")
print(ai_dataset.job_title.value_counts())

print(ai_dataset[ai_dataset.job_title == "Head of AI"].company_size.value_counts())

#print(pd.to_datetime(ai_dataset.posting_date))

ai_dataset.posting_date = pd.to_datetime(ai_dataset.posting_date)
ai_dataset.application_deadline = pd.to_datetime(ai_dataset.application_deadline)

print(ai_dataset.posting_date)
print(ai_dataset.application_deadline)

ai_dataset["job_position_closes_in:"] = ai_dataset.application_deadline - ai_dataset.posting_date
print(ai_dataset["job_position_closes_in:"])
print(ai_dataset["job_position_closes_in:"].value_counts())

print(ai_dataset["job_position_closes_in:"].dt.days.describe())

print(ai_dataset.required_skills)
print(ai_dataset.required_skills.str.split(", "))
print(ai_dataset.required_skills.str.split(", ", expand = True))

ai_dataset_skills = ai_dataset.required_skills.str.split(", ", expand = True)

print(ai_dataset_skills)
print(ai_dataset_skills[1] == "Deep Learning")
print(ai_dataset_skills[ai_dataset_skills[1] == "Deep Learning"])
print("------------------------------------------------------------------------------")
print(ai_dataset.loc[ai_dataset_skills[ai_dataset_skills[1] == "Deep Learning"].index])
print("------------------------------------------------------------------------------")

print("let see")
print(ai_dataset[ai_dataset.required_skills.str.contains("Deep Learning", case = False)])
print("--------------------------------------------------------------------------------")


plt.hist(ai_dataset["job_position_closes_in:"].dt.days, bins = "fd")
plt.show()

