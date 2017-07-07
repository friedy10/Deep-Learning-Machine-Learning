import random

with open("england_junk.csv") as fr:
    with open("england_test.csv", "a") as f1, open("england_validation.csv", "a") as f2:
        for line in fr.readlines():
            rd = random.randint(1, 2)
            f = f1 if rd == 1 else f2
            f.write(line)
