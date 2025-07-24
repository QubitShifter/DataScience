from Utils.functions.func_parse_trimesters import parse_trimester
from Utils.dicts.dict_Month_Quarters import months_Quarter


def test_quarters():
    for quarter in months_Quarter:
        result = parse_trimester(quarter)
        print(f"Input: {quarter} -> Output: {result}")


print(test_quarters())
