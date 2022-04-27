import sys
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

#Se lee fichero de interests.txt por línea de comandos
with open(sys.argv[1], 'r') as f:
    liked = [line.rstrip() for line in f]

#Se lee fichero de recommendations.txt por línea de comandos
with open(sys.argv[2], 'r') as f:
    recommend = [line.rstrip() for line in f]

#Extraemos documentos que sabemos que le han gustado al usuario
interests = list(map(lambda x: re.split(r".\s", x, 1), liked))

dfliked = pd.DataFrame(interests, columns=['DocNumb', 'Document'])

vectorizer = TfidfVectorizer(stop_words = "english")
tfidfmatrixint = vectorizer.fit_transform(dfliked['Document'])
words = vectorizer.get_feature_names_out()
#En vocabulario se guardan los términos y su orden..
# vocabulary = vectorizer.vocabulary_
# transformer = TfidfTransformer()

recommendations = list(map(lambda x: re.split(r".\s", x, 1), recommend))
dfrecommend = pd.DataFrame(recommendations, columns=['DocNumb', 'Document'])

#vectorizer = CountVectorizer(stop_words = "english")
tfidfmatrixrec = vectorizer.transform(dfrecommend['Document'])

def recommend():
    #Hacer función recomendación
    print("")

#Se calcula la similitud del coseno: Se calcula cuánto de similares son los documentos. 
#Cuando sale un 1 es porque se esta calculando la similitud de un documento consigo mismo.
#Con [0:1] se hace la similitud del coseno de los vectores con el primer documento.
cosine_similarities = cosine_similarity(tfidfmatrixrec, tfidfmatrixint)
print(cosine_similarities)

fila = 0
columna = 0
for idi, i in enumerate(cosine_similarities):
    for j in range(0, len(i)):
        if i[j] > 0.8:
            fila = idi
            columna = j
            print(j) #Columnas
            print(idi) #Filas

print(cosine_similarities[idi][j])
        

recommend()