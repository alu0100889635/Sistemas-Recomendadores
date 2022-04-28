import sys
from cv2 import sort
import pandas as pd
import re
import numpy as np
from requests import delete
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

test = []

for i in documents:
    test.append([i[1]])

words = []
tfidfdoc = []
positions = []
exes = []

for i in test:
    vectorizer = CountVectorizer(stop_words = "english")
    X = vectorizer.fit_transform(i)
    word = vectorizer.get_feature_names_out()
    position = []
    for x in word:
        position.append(i[0].lower().find(x))
    positions.append(position)
    words.append(word)
    exes.append(X.toarray())
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    tfidfdoc.append(tfidf.toarray())

dfdocs = pd.DataFrame(documents, columns=['DocNumb', 'Document'])
dfdocs['Term Ind'] = positions
dfdocs['Terms'] = words
dfdocs['TF-IDF'] = tfidfdoc
dfdocs['Like'] = [1, 1, 1, 1, 0, 0, 0, 0, 0]
print(dfdocs)

#dfdocs = pd.DataFrame(documents, columns=['DocNumb', 'Document'])
#Se añade columna al df para saber qué textos le han gustado y que textos aún no se sabe si le ahn gustado o no"

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


dfmatrix = pd.DataFrame(lowerTriangleMatrix)
sorted_values = pd.Series

for i in range(0, len(dfmatrix.columns)):
    if dfdocs.iloc[i]['Like'] == 1:
        print("\nPorque te ha gustado el documento ", dfdocs.iloc[i]['DocNumb'] ,", te recomendamos: ")
        sorted_values = dfmatrix[i].sort_values(ascending = False)
        for index, value in sorted_values.items():
            if(round(value, 6) != 1.0):
                if(round(value, 6) != 0.0):
                    print("Documento ", dfdocs.iloc[index]['DocNumb'],  " -> Similitud con documento", dfdocs.iloc[i]['DocNumb'], "= ", round(value, 6))
    