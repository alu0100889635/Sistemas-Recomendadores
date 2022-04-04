import sys
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

#Se lee fichero de recommendations.txt por línea de comandos
with open(sys.argv[1], 'r') as f:
    recommend = [line.rstrip() for line in f]

# #Se lee fichero de recomendar.txt por línea de comandos
# with open(sys.argv[2], 'r') as f:
#     recommend = [line.rstrip() for line in f]

#Extraemos documentos que vamos a ver si podemos recomendar
# recomendations = list(map(lambda x: re.split(r".\s", x, 1), recommend))
# dfrec = pd.DataFrame(recomendations, columns=['DocNumb', 'Document'])

#Extraemos documentos que sabemos que le han gustado al usuario
recommendations = list(map(lambda x: re.split(r".\s", x, 1), recommend))
test = []

for i in recommendations:
    test.append([i[1]])

#Se calculan los vectores TF-IDF para cada término del documento
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

dfrecommend = pd.DataFrame(recommendations, columns=['DocNumb', 'Document'])
dfrecommend['Term Ind'] = positions
dfrecommend['Terms'] = words
dfrecommend['TF-IDF'] = tfidfdoc

print(dfrecommend)