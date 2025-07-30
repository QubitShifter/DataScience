import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


a, b = 2, 3
x = np.linspace(-5, 5, 100)
x = np.random.choice(x, size=20)
y =a*x+b

plt.scatter(x, y)
plt.xlabel("x; independant variable")
plt.ylabel("y; dependant variable")
plt.show()

y_noice = np.random.normal(loc = 0, scale =1 , size= len(x))
y += y_noice

plt.scatter(x, y)
plt.xlabel("x; independant variable")
plt.ylabel("y; dependant variable")
plt.show()


y = a*x+b + y_noice
plt.scatter(x, y)
plt.xlabel("x; independant variable")
plt.ylabel("y; dependant variable")
plt.show()