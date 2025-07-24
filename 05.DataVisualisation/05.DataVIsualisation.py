import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import scipy.stats as st



pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.precision', 2)

EV_dataset = pd.read_csv("lab/electric_vehicles_specs.csv")

print(EV_dataset.shape)
print(EV_dataset.columns)
print(EV_dataset.describe().T)
print(EV_dataset.dtypes)
print(EV_dataset.segment)
print(EV_dataset.segment.value_counts())
print(EV_dataset.brand.value_counts())
ev_brands = EV_dataset.brand.value_counts()

#
# plt.figure(figsize= (6, 12))
# #plt.bar(ev_brands.index, ev_brands)
# plt.barh(ev_brands.index, ev_brands)
# #plt.barh(ev_brands.index, ev_brands / len(ev_brands)) - ## frequency
# plt.xlabel("Number of Cars")
# plt.ylabel("Brand")
# plt.show()

print(EV_dataset.battery_type.value_counts())
  # drop column "battery_type" because ity cointains same value and has low entropy
print(EV_dataset.dtypes)
print(EV_dataset.fast_charge_port.value_counts(dropna=False))
print(EV_dataset[EV_dataset.fast_charge_port.isna()].source_url)
print(EV_dataset[EV_dataset.fast_charge_port.isna()].source_url.values[0])
print(EV_dataset[EV_dataset.fast_charge_port == "CHAdeMO"].source_url.values[0])
print(EV_dataset.info()) # you can compare if each value has some variable
print(EV_dataset.cargo_volume_l.value_counts())
print(EV_dataset.cargo_volume_l.unique())
#print(EV_dataset[EV_dataset.cargo_volume_l.fillna("").str.contains("banana", case = False)])

##print(EV_dataset.cargo_volume_l.copy())
##EV_dataset["Cargo_volume_corrected"] = EV_dataset.cargo_volume_l.copy()
#indices_to_correct = EV_dataset[EV_dataset.cargo_volume_l.fillna("").str.contains("banana", case = False)].index

print(EV_dataset[EV_dataset.cargo_volume_l.fillna("").str.contains("banana", case = False)].index) # питам дали "cargo_volume_l" съдържа банана боксес
                                                                                            # и им взимаме индексите

# #ще си направим нова колона
# EV_dataset["cargo_volume_corrected"] = EV_dataset.cargo_volume_l.copy()
# indices_to_correcr = EV_dataset[EV_dataset.cargo_volume_l.fillna("").str.contains("banana", case = False)].index
# print(EV_dataset.loc[indices_to_correcr, "cargo_volume_corrected"])
# EV_dataset.loc[indices_to_correcr, "cargo_volume_corrected"] = np.nan # най-добрия начин за коригиране на данни, с loc, index и колона,
#                                                                       # като индекса идва най-добре от някаква маска -> EV_dataset.cargo_volume_l.fillna("").str.contains("banana", case = False)]

# base_value = 855.1688 / 10
# EV_dataset.loc[indices_to_correcr, "cargo_volume_corrected"] = [base_value * 10, base_value * 31, base_value * 13]

#print(EV_dataset.loc[indices_to_correcr, "cargo_volume_corrected"].round())


#print(EV_dataset["cargo_volume_corrected"])
#print(EV_dataset.cargo_volume_l.astype(float, errors="ignore")) # търсом останалите от банана бокс, които не са липсващи, взимаме
                                                                # стойността на колоните, обръшаме я във float, и подтискаме грешките



# print (EV_dataset.loc[indices_to_correcr, "cargo_volume_corrected"])
# print(EV_dataset.cargo_volume_corrected.astype(float))
# EV_dataset.cargo_volume_corrected = EV_dataset.cargo_volume_corrected.astype(float)

# plt.hist(EV_dataset.cargo_volume_corrected, bins=25)
# plt.axvline(855.1688, c="r")

print(EV_dataset.describe())
print(EV_dataset.describe().T)   # показва само числени колони
print(EV_dataset.segment)
print(EV_dataset.segment.unique())
print(EV_dataset.segment.value_counts())
print(EV_dataset.brand.value_counts())
brands = EV_dataset.brand.value_counts()
print("------------------------------------")


#plt.figure(figsize=(10, 15))
# plt.barh(brands.index, brands / len(brands) * 100)
# plt.xlabel("Frequency of cars [%]")
# plt.ylabel("Brand")
# plt.show()

print("------------------------------------")
# да видим типовете батерии, - излизат, че всички са еднакви => можем да дропнем колоната (еднакви записи => ентропията е 0. Не ни трябват еднакви променливи)
print(EV_dataset.battery_type.value_counts())
EV_dataset = EV_dataset.drop(columns= ["battery_type"])
print(EV_dataset.columns)
print("------------------------------------")
print(EV_dataset.model.unique())
print("------------------------------------")
print(EV_dataset.model.value_counts(dropna=False))
print("------------------------------------")
print(EV_dataset.fast_charge_port.value_counts(dropna=False))
#имаме почти цял дейта сет в тази колона с една и съща стойност. искаме да видим каква точно е тя
print(EV_dataset[EV_dataset.fast_charge_port.isna()])
print(EV_dataset[EV_dataset.fast_charge_port.isna()].source_url)# така показва урл-а от колоната 'source_url' с точки и не можем да го копираме
print(EV_dataset[EV_dataset.fast_charge_port.isna()].source_url.values[0])

# можвм да проверим една променлива как е разпределена по колони. с .info()
print(EV_dataset.info())
print("--------------------------------------")

print(EV_dataset.cargo_volume_l.value_counts())
print(EV_dataset.cargo_volume_l.unique())

#да извикаме само тези които имат обем измерен в банана боксес

print(EV_dataset[EV_dataset.cargo_volume_l.fillna("").str.contains("banana", case = False)])

#да пробваме да обърнем колоната в флоут
# 1во взимаме индексите
EV_dataset["cargo_volume_corrected"] = EV_dataset.cargo_volume_l.copy()
indices_to_correct = EV_dataset[EV_dataset.cargo_volume_l.fillna("").str.contains("banana", case = False)].index
print(EV_dataset.loc[indices_to_correct, "cargo_volume_corrected"])
EV_dataset.loc[indices_to_correct, "cargo_volume_corrected"] = np.nan  #това е начина за коригиране на стойности в данните.. използва се loc, индекс и колона
                                                                       # индекса е най-добре да дойде от някаква маска, етп това филтриране се нарича маска
                                                                       # EV_dataset.cargo_volume_l.fillna("").str.contains("banana", case = False)
                                                                       # имаме две операции в един ред -> заменяме nan с празен стринг, и търсим "bananas"

print("----------------------------------------------------")
print("----------------------------------------------------")
print(EV_dataset.loc[indices_to_correct, "cargo_volume_corrected"])
print(EV_dataset[EV_dataset.cargo_volume_l.fillna("").str.contains("banana", case = False)].index)

print(EV_dataset.cargo_volume_l.astype(float, errors='ignore'))
print(EV_dataset.cargo_volume_corrected.astype(float))   # тък вече колоната ни е float64

#презаписваме я
EV_dataset.cargo_volume_corrected = EV_dataset.cargo_volume_corrected.astype(float)

base_value = 855.1688 / 10
EV_dataset.loc[indices_to_correct, "cargo_volume_corrected"] = [base_value * 10, base_value * 31, base_value * 13]
print(EV_dataset.loc[indices_to_correct, "cargo_volume_corrected"].round())


#Можем да присвоим стойностите обратно
EV_dataset.loc[indices_to_correct, "cargo_volume_corrected"] = EV_dataset.loc[indices_to_correct, "cargo_volume_corrected"].round()
# вземаме данните на редовете 'indices_to_correct' и на колоната "cargo_volume_corrected" и ги приравни на ->
# .loc ни дава референция към стойностите, само с него може да се редактира дейта фрейм


plt.hist(EV_dataset.cargo_volume_corrected, bins = 25)

plt.axvline(855.1688, c = "r")  # искаме да покажем къде точно на хистограмата се намират литрите за конкретния автомобил които гледаме
                                # т.е сложихме число което не ни говори нищо в някакъв контекс.

plt.xlabel("Volume [l]")
plt.ylabel("Count")
plt.show()

#1. виждаме, че имаме мултимодално разпределение, с много голям пик.
# Мултимодалност означава, че имаме най-вероятно някакъв системен байъс
# plt.xlabel("Volume [l]")
# plt.ylabel("Count")
# plt.show()


# Понягога е грешно да имаш хистограма. Хистограмите в matplotlib са направени за непрекъснати променливи.
# Ако плотнем категорийна променлива, биновете ще се размибават с центровете си(защото стойностите не са непрекъснати, а имат
# дискретен характер.)
# как се прави хистограма на категорийна променлива. ->  показваме всяка една стойност поотделно


plt.hist(EV_dataset.seats, bins = 20)
plt.show()

# как се прави хистограма на категорийна променлива. ->  показваме всяка една стойност поотделно
print(EV_dataset.seats.unique())
print(EV_dataset.seats.value_counts())


print(EV_dataset.seats.value_counts())
plt.bar(EV_dataset.seats.value_counts().index, EV_dataset.seats.value_counts())
plt.show()
#сега вече виждаме индексите и стойностите им и можем с тях да направим barchar


plt.bar(EV_dataset.seats.value_counts().index, EV_dataset.seats.value_counts())
plt.title("The majority of cars are 5-seaters")
plt.xlabel("Number of seats")
plt.ylabel("Count")
plt.show()


plt.hist(EV_dataset.length_mm)
plt.hist(EV_dataset.width_mm)
plt.hist(EV_dataset.height_mm)
plt.show()

EV_dataset["volume_proxy"] = EV_dataset.length_mm * EV_dataset.width_mm * EV_dataset.height_mm
plt.hist(EV_dataset.volume_proxy / (100 ** 3), bins=20)
plt.xlabel("Volume, $[l]$")
plt.show()


# def _segment(EV_dataset):
#     segments_list = []
#     segment_names = []
#     for segment, group_data in EV_dataset.groupby("segment"):
#         print(segment, len(group_data))
#         segments_list.append(group_data.efficiency_wh_per_km)
#         segment_names.append(segment)
#     return segments_list, segment_names

def _segment(EV_dataset):
    segments = {}
    for segment, group_data in EV_dataset.groupby("segment"):
        segments[segment] = group_data.efficiency_wh_per_km.values
        print(f"{segment}: {len(group_data)} records")
    return segments

segments = _segment(EV_dataset)
#print(_segment(segments_list))

plt.boxplot(segments.values(), labels=segments.keys())
plt.xticks(rotation=45)
plt.ylabel("Efficiency (Wh/km)")
plt.title("Efficiency by Segment")
plt.tight_layout()
plt.show()



drive_trains = {}
for drive_train, group_data in EV_dataset.groupby("drivetrain"):
    drive_trains[drive_train] = group_data.efficiency_wh_per_km.values

plt.boxplot(drive_trains.values(), labels=drive_trains.keys())
plt.xticks(rotation=45)
plt.ylabel("Efficiency (Wh/km)")
plt.title("Efficiency by Segment")
plt.tight_layout()
plt.show()

### Връски между променливи
## Да сравняваме неща спрямо категории

# Число спрямо категория

# Число спрямо число -> scatter plot

plt.scatter(EV_dataset.torque_nm, EV_dataset.range_km)
plt.xlabel("Torque [N.m]")
plt.ylabel("Range [km]")
plt.semilogx()
plt.show()

colors = EV_dataset.drivetrain.replace({"AWD": 0, "FWD": 1, "RWD": 2})
plt.scatter(EV_dataset.torque_nm, EV_dataset.range_km, c = colors, s=7, marker = "x")
plt.xlabel("Torque [N.m]")
plt.ylabel("Range [km]")
#plt.semilogx()
plt.show()

plt.scatter(EV_dataset.efficiency_wh_per_km, EV_dataset.acceleration_0_100_s, c = colors)
plt.xlabel("Efficiency [Wh/km]")
plt.ylabel("Acceleration 0-100 [s]")

plt.show()