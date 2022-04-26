import sys
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

#Se lee fichero de interests.txt por línea de comandos
with open(sys.argv[1], 'r') as f:
    liked = [line.rstrip() for line in f]

#Extraemos documentos que sabemos que le han gustado al usuario
interests = list(map(lambda x: re.split(r".\s", x, 1), liked))

dfliked = pd.DataFrame(interests, columns=['DocNumb', 'Document'])

vectorizer = CountVectorizer(stop_words = "english")
X = vectorizer.fit_transform(dfliked['Document'])
words = vectorizer.get_feature_names_out()
#En vocabulario se guardan los términos y su orden..
vocabulary = vectorizer.vocabulary_
transformer = TfidfTransformer()
tfidfmatrix = transformer.fit_transform(X)

def recommend():
    #Hacer función recomendación
    print("")


#Se calcula la similitud del coseno
#Con [0:1] se hace la similitud del coseno de los vectores con el primer documento.
#cosine_similarities = cosine_similarity(tfidfmatrix[0:1], tfidfmatrix).flatten()
cosine_similarities = cosine_similarity(tfidfmatrix, tfidfmatrix)
print(cosine_similarities)

recommend()

# cosine_similarities = list(enumerate(map(cosine_similarity, dfliked['TF-IDF'])))
# cosine_similarities = sorted(cosine_similarities, key = lambda x: x[1], reverse = True)
# cosine_similarities = cosine_similarities[1:6]
# doc_indices = [i[0] for i in cosine_similarities]
# rec = dfliked.iloc[doc_indices]

# for index, row in rec.iterrows():
#     print(row['DocNumb'], ".", row['Document'])