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

dflikes = dfdocs.loc[dfdocs.Like == 1]
dfrec = dfdocs.loc[dfdocs.Like != 1]

vectorizer = TfidfVectorizer(stop_words = "english")
tfidfmatrixlike = vectorizer.fit_transform(dfdocs['Document'])
words = vectorizer.get_feature_names_out()

tfidfmatrixrec = vectorizer.transform(dfrec['Document'])

cosine_similarities = cosine_similarity(tfidfmatrixrec, tfidfmatrixlike)
print(cosine_similarities)


fila = []
for idi, i in enumerate(cosine_similarities):
    for j in range(0, len(i)):
        if i[j] >= 0.8 and i[j]<1:
            fila.append(idi)
        
print("\nTextos a recomendar: \n")
for x in fila:
    print(dfrec.iloc[x]['DocNumb'], ". ", dfrec.iloc[x]['Document'])