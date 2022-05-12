import sys
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Se lee fichero de textos.txt por línea de comandos
with open(sys.argv[1], 'r') as f:
    liked = [line.rstrip() for line in f]

#Extraemos documentos que sabemos que le han gustado al usuario
documents = list(map(lambda x: re.split(r".\s", x, 1), liked))
dfdocs = pd.DataFrame(documents, columns=['DocNumb', 'Document'])

#Se añade columna al df para saber qué textos le han gustado y que textos aún no se sabe si le ahn gustado o no"
dfdocs['Like'] = [1, 1, 1, 1, 0, 0, 0, 0, 0]

#Se añade al df columnas relativas a los términos de cada documento, al índice de cada término y al tf-idf de cada documento
test = []
for i in documents:
    test.append([i[1]])

def vectorizeDocs(doc):
    vectorizer = TfidfVectorizer(stop_words = "english")
    return vectorizer.fit_transform(doc), vectorizer.get_feature_names_out()

#Se calcula el tfidf de todos los documentos y se sacan los términos
tfidfmatrix, words = vectorizeDocs(dfdocs['Document'])

#Se guarda en un df el tf-idf de cada uno de los términos de cada doc
tfidfdoc = []
for i in tfidfmatrix.toarray():
    tfidfdoc.append(i)
# for i in tfidfmatrix.toarray():
#     tfidfdocnonzero = []
#     for x in range(0, len(i)):
#         if(i[x]) != 0.0:
#             tfidfdocnonzero.append(i[x])
#     tfidfdoc.append(tfidfdocnonzero)

# Se guarda en ese mismo df los términos y los índices de esos términos
words = []
positions = []
for i in test:
    X, word = vectorizeDocs(i)
    position = []
    for w in word:
        position.append(i[0].lower().find(w))
    positions.append(position)
    words.append(word)

dfdocs['TF-IDF'] = tfidfdoc
dfdocs['Term Ind'] = positions
dfdocs['Terms'] = words
print(dfdocs[['DocNumb', 'Term Ind', 'Terms', 'TF-IDF']])


# #Se calcula la similitud del coseno: Se calcula cuánto de similares son los documentos. 
# #Cuando sale un 1 es porque se esta calculando la similitud de un documento consigo mismo.

cosine_similarities = cosine_similarity(tfidfmatrix, tfidfmatrix)
lowerTriangleMatrix = np.tril(cosine_similarities)

fila = []
for idi, i in enumerate(lowerTriangleMatrix):
    for j in range(0, len(i)):
        if 0.8 <= round(i[j], 6) < 1.0:
            fila.append(idi)


#Se ordenan los documentos de mayor a menor similitud y se le muestran al usuario según los docs que le han gustado previamente
dfmatrix = pd.DataFrame(lowerTriangleMatrix)
sorted_values = pd.Series

for i in range(0, len(dfmatrix.columns)):
    if dfdocs.iloc[i]['Like'] == 1:
        print("\nPorque te ha gustado el documento ", dfdocs.iloc[i]['DocNumb'] ,", te recomendamos: ")
        sorted_values = dfmatrix[i].sort_values(ascending = False)
        for index, value in sorted_values.items():
            if(round(value, 6) != 1.0): #Se cogen los valores distintos de uno que hay en la matriz triangular
                if(round(value, 6) != 0.0): #Se cogen los valores distintos de cero que hay en la matriz triangular
                    if(dfdocs.iloc[index]['Like'] != 1): #Se recomiendan solo los que no le han gustado aún
                        print("Documento ", dfdocs.iloc[index]['DocNumb'],  " -> Similitud con documento", dfdocs.iloc[i]['DocNumb'], "= ", round(value, 6))
    