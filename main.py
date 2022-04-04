import sys
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

#Se lee fichero de interests.txt por línea de comandos
with open(sys.argv[1], 'r') as f:
    liked = [line.rstrip() for line in f]

#Se lee fichero de recomendar.txt por línea de comandos
with open(sys.argv[2], 'r') as f:
    recommend = [line.rstrip() for line in f]

#Extraemos documentos que vamos a ver si podemos recomendar
recomendations = list(map(lambda x: re.split(r".\s", x, 1), recommend))
dfrec = pd.DataFrame(recomendations, columns=['DocNumb', 'Document'])

#Extraemos documentos que sabemos que le han gustado al usuario
interests = list(map(lambda x: re.split(r".\s", x, 1), liked))
test = []

for i in interests:
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

dfliked = pd.DataFrame(interests, columns=['DocNumb', 'Document'])
dfliked['Term Ind'] = positions
dfliked['Terms'] = words
dfliked['TF-IDF'] = tfidfdoc

print(dfliked)

cosine_similarities = list(enumerate(map(cosine_similarity, dfliked['TF-IDF'])))
# cosine_similarities = sorted(cosine_similarities, key = lambda x: x[1], reverse = True)
# cosine_similarities = cosine_similarities[1:6]

print(cosine_similarities)