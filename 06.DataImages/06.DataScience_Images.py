import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.stats as st
from skimage.io import imread, imshow

print(os.listdir("images/"))


with open("images/cat.jpg", "rb") as f:
    content = f.read()
print(content)

cat_image = imread("images/cat.jpg")

print(cat_image)

print(type(cat_image))
print(cat_image.shape)
print(cat_image.dtype)
print(cat_image.min(), cat_image.max())

print(cat_image[0, 0])
print(cat_image[0, 0, :])
print(cat_image[0, :, :])
print(len(cat_image[0, :, :]))

print(cat_image[:, 0, :])
print(len(cat_image[:, 0, :]))

print(cat_image[:, :, 0])
print(cat_image[:, :, 0].shape)


f, axes = plt.subplots(1, 3, figsize = (10, 13))
print(axes)

axes[0].imshow(cat_image[:, :, 0], cmap="Reds")
axes[1].imshow(cat_image[:, :, 1], cmap="Greens")
axes[2].imshow(cat_image[:, :, 2], cmap="Blues")
#axes[3].imshow(cat_image[:, :, 3], cmap="gray")
plt.tight_layout()
# red_channel = cat_image[:, :, 0]
# plt.imshow(cat_image)
# plt.imshow(red_channel, cmap="Reds")
# plt.colorbar()
plt.show()

red_channel, green_channel, blue_channel = cat_image[:, :, 0], cat_image[:, :, 1], cat_image[:, :, 2]
print(red_channel)
print(red_channel.shape)

print(red_channel[20, :])
print(red_channel[20, :].mean())


# средния интензитет на светлината който е попаднал в червения канал
mean_intensity_rch = red_channel.mean()
print(f"средният интензитет за червения канал е: {mean_intensity_rch:.2f}")
print(red_channel.mean(axis = 0).shape)
print(red_channel.mean(axis = 1).shape)

# \sigma -> средното разстояние от \mu
sigma_red_channel = red_channel.std(ddof = 1)
print(f"стандартното отклонение за червеният канал е: {sigma_red_channel:.2f}")

### колко пъти средния сигнал е по-голям от типичното отклонение за него - ( колко пъти полезния сигнал е по-голям от шума)
### мярка за контраст
snr = mean_intensity_rch / sigma_red_channel

print(f"signal to noice ration e {snr:.3f}")

plt.figure(figsize=(10, 7))
plt.bar(range(len(red_channel.mean(axis=1))), red_channel.mean(axis=1)) # диаграма на средното количество червено за всяка една колона, не е хистограма
plt.xlim(0, 640)
plt.show()


### ако ио

print(' ' * 48)
random_image = np.random.randint(0, 255, (100,100))
print(random_image)

plt.imshow(random_image, cmap="gray")
plt.show()

plt.hist(red_channel)  # red_channel e двумерен масив, numpy прави хистограма за всеки един ред поотделно, това което ни трябва е да превърнем red_channel в едномерен масив
red_channel.ravel()   # вземаме първия ред, до него долепяме втория, до него третие и т.н.. така става един много голям вектор -> rеshape() ийли ravel()
red_channel_1d = red_channel.ravel()
green_channel_1d = green_channel.ravel()
blue_channel_1d = blue_channel.ravel()
print(red_channel_1d)
plt.hist(red_channel_1d, bins=256, color="red")
plt.show()

# Можем да покажем трите канала едновремено, тоест трите инстанции/канали  с f:

f, axes = plt.subplots(1, 3, figsize = (14, 3))

axes[0].hist(red_channel_1d, bins=256, color="r")
axes[1].hist(green_channel_1d, bins=256, color="g")
axes[2].hist(blue_channel_1d, bins=256, color="b")

# по-удобно е да покажем хистограмата на едно и също място
plt.hist(red_channel_1d, bins=256, color="r", alpha=0.5)
plt.hist(green_channel_1d, bins=256, color="g", alpha=0.5)
plt.hist(blue_channel_1d, bins=256, color="b", alpha=0.5)

plt.xlim(0, 255) # задаваме начална и крайна стойност на хистограмата
plt.show()

red_copy = red_channel.copy()
#plt.imshow(red_copy[100:450, 400:750]) # избираме си конкретна част от изображението
red_copy[100:450, 400:750] = 0 # маскираме изображението в избрания регион, 0 - черна маска, 255 - бяла
plt.imshow(red_copy, cmap="gray")
plt.show()

print(red_copy[red_copy < 20]) # искаме да покабем много тъмните цветопве

# и ги направи нули
red_copy[red_copy < 20] = 0
red_copy[red_copy > 235] = 255

plt.imshow(red_channel, cmap="gray", vmin=0, vmax=255)
plt.show()
plt.imshow(red_copy, cmap="gray", vmin=0, vmax=255)
plt.show()

#това се нарича клипинг, да променим стойнистите на пикслели с определена стойност
#съответно хистограмата се променя също..

plt.hist(red_copy.ravel(), bins=255, cumulative=True)
plt.show()

red_channel = 255 - red_channel
plt.imshow(red_channel, cmap="gray")
plt.show()