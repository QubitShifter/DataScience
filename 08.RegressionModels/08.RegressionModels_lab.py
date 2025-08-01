import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Utils.Functions.func_helper_print_colors import color_print

#a, b = 2, 3
a, b = np.random.uniform(-5, 5, size=2)
a_real, b_real = a, b
print()

print(a, b)
x = np.linspace(-5, 5, 100)
x = np.random.choice(x, size=30)
y =a * x + b
print(y)

y_noice = np.random.normal(loc = 0, scale =1.5 , size= len(x))
y =a * x + b + y_noice

pd.DataFrame({
    "x": x,
    "y": y
}).to_csv("noice_example.csv", index=None)

noice_dataframe = pd.read_csv("noice_example.csv")
print(noice_dataframe)

a = 10
b = 5
model = a * x + b

#
# plt.scatter(x, y)
# plt.xlabel("x; independant variable")
# plt.ylabel("y; dependant variable")
# plt.show()

color_print(
    f"""
да видим какво представлява шума
     """
    , level="info"
)


# print(y_noice)
# print(y_noice.mean())
# color_print(f"добавяме 'y_noice' към ax+b")
# #y += y_noice # така всеки път ще добавя и презаписва новите стойности за това го пишем както е долу:
y = a * x + b + y_noice

color_print(
    f"""
можем да си поиграем с scale което е стандартното отклонение 
и да видим как се променя шума  

     """
    , level="info"
)


color_print(
    f"""
в случая шума е само по y, нагоре и надолу, по x няма шум, 
за сега предполагаме, че няма 
     """
    , level="info"
)

color_print(
    f"""
колкото по-голямо е стандартното отклонение, толкова повече полезния сигнал изчезва  
спрямо шума
     """
    , level="info"
)


color_print(
    f"""
за да придобием някаква представа колко е влиянието на стандартното отклонение, 
можем да вземем някакъв рейндж от 30 до -30 и виждаме каква част от него 
е  стандартното отклонение.. ( ако stdin e 20, и имаме интервал от -30 do 30, то 20 е 1/3 от 60)
     """
    , level="info"
)
plt.scatter(x, y)
plt.xlabel("x; independant variable")
plt.ylabel("y; dependant variable")
plt.show()

color_print(
    f"""
да умножим всичко по адитивния шум
     """
    , level="info"
)

color_print(
    f"""
ще видим, че влиянието му е доста по-голямо от това да съберем с него 
     """
    , level="info"
)

color_print(
    f"""
този вид шум може да бъде направен адитивен с логаритъм
 y = (a*x+b) * y_noice  -> ако вземем логъритъм от двете страни..
 логъритъм от произведение е сбор от логаритми
 ln(y) = ln(a*x+b) + ln(y_noice) 
     """
    , level="info"
)


#y = (a*x+b) * y_noice # това е адитивен шум. шум който се добавя и всяко едно число y е отместено с повече или по-малко
plt.scatter(x, y)
plt.xlabel("x; independant variable")
plt.ylabel("y; dependant variable")
plt.show()

color_print(
    f"""
какво би станало ако шумът зависеше от данните??? 
колкото по-голямо е числото x, толкова по-голям става шума
където по оста X имаме -4,  шума ще се умножи по -4 и ще отиде още по-надолу,
там където имаме 4, шума ще се умножи по 4 и ще отиде още по-нагоре
     """
    , level="info"
)

# y_noice = np.random.normal(loc = 0, scale =1.5 , size= len(x)) * x
# #
# # y = (a*x+b) * y_noice # това е адитивен шум. шум който се добавя и всяко едно число y е отместено с повече или по-малко
# # plt.scatter(x, y)
# # plt.xlabel("x; independant variable")
# # plt.ylabel("y; dependant variable")
# # plt.show()




color_print(
    f"""
гледайки графиката можем да предполагаме, че имаме линейна зависимост.
имаме данни закоито можем да знаем нещо или да не знаем нищо.
как да си направим функция която да ги предиктва
     """
    , level="info"
)

color_print(
    f"""
нека пробваме различни модели по метода на тестването 
с различни стойности за a и b
     """
    , level="info"
)



color_print(
    f"""
можем да наложим модела върху данните
     """
    , level="info"
)


plt.plot(x, model, c = "r")
plt.show()


def display_model_result(data, a, b):
    plt.scatter(data.x, data.y)
    plt.xlabel("x")
    plt.ylabel("y")
    model = a * x + b
    plt.plot(data.x, model, c ="r", label= f"{a:.2f}x + {b:.2f}")

    plt.legend()
    plt.show()


print(display_model_result(noice_dataframe, 8, 6))
print(display_model_result(noice_dataframe, 12, -6))
print(display_model_result(noice_dataframe, 10, 5))
print(display_model_result(noice_dataframe, 10.346, 5.1712))
print(display_model_result(noice_dataframe, 4.1, 6))
