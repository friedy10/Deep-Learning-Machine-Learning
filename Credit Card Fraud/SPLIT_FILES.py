import random

with open("creditcard_junk.csv") as fr:
    with open("creditcard_validation.csv", "a") as f1, open("creditcard_testing.csv", "a") as f2:
        for line in fr.readlines():
            rd = random.randint(1, 2)
            f = f1 if rd == 1 else f2
            f.write(line)
