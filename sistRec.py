import sys
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

#Se lee fichero de textos.txt por línea de comandos
with open(sys.argv[1], 'r') as f:
    liked = [line.rstrip() for line in f]

#Extraemos documentos que sabemos que le han gustado al usuario
documents = list(map(lambda x: re.split(r".\s", x, 1), liked))

dfdocs = pd.DataFrame(documents, columns=['DocNumb', 'Document'])
#Se añade columna al df para saber qué textos le han gustado y que textos aún no se sabe si le ahn gustado o no"
dfdocs['Like'] = [1, 1, 1, 1, 0, 0, 0, 0, 0]
print(dfdocs)

# dflikes = dfdocs.loc[dfdocs.Like == 1]
# dfrec = dfdocs.loc[dfdocs.Like != 1]

vectorizer = TfidfVectorizer(stop_words = "english")
tfidfmatrix = vectorizer.fit_transform(dfdocs['Document'])
words = vectorizer.get_feature_names_out()

# #Se calcula la similitud del coseno: Se calcula cuánto de similares son los documentos. 
# #Cuando sale un 1 es porque se esta calculando la similitud de un documento consigo mismo.

cosine_similarities = cosine_similarity(tfidfmatrix, tfidfmatrix)
#print(cosine_similarities)

lowerTriangleMatrix = np.tril(cosine_similarities)

# fila = []
# lowerTriangleMatrix = lowerTriangleMatrix.flatten()
# sorted = np.sort(lowerTriangleMatrix)[::-1]
# for idi, i in enumerate(sorted):
#         if 0.0 < round(i, 6) < 1.0:
#             fila.append(idi)

fila = []
for idi, i in enumerate(lowerTriangleMatrix):
    for j in range(0, len(i)):
        if 0.8 <= round(i[j], 6) < 1.0:
            fila.append(idi)


dfmatrix = pd.DataFrame(lowerTriangleMatrix, columns=['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9'])
#print(dfmatrix)

print("\nTextos a recomendar: \n")
for x in fila:
    if dfdocs.iloc[x]['Like'] != 1:
        print(dfdocs.iloc[x]['DocNumb'], ". ", dfdocs.iloc[x]['Document'])