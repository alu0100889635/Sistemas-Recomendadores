import sys
import pandas as pd
import re

#Se lee fichero por línea de comandos
with open(sys.argv[1], 'r') as f:
    contents = [line.rstrip() for line in f]

foo = list(map(lambda x: re.split(r".\s", x, 1), contents))

for i in contents:
    x = re.split(r".\s", i, 1)

df = pd.DataFrame(foo, columns=['DocNumb', 'Document'])

df["Ind"] = " "
df["Term"] = " "
df["TF"] = " "
df["TF-IDF"] = " "
df["IDF"] = " "
print(df)