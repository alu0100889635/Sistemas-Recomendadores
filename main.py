import sys
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

#Se lee fichero por línea de comandos
with open(sys.argv[1], 'r') as f:
    contents = [line.rstrip() for line in f]

foo = list(map(lambda x: re.split(r".\s", x, 1), contents))

test = []

for i in foo:
    test.append([i[1]])


words = []
tfidfdoc = []
positions = []

for i in test:
    vectorizer = CountVectorizer(stop_words = "english")
    X = vectorizer.fit_transform(i)
    word = vectorizer.get_feature_names_out()
    position = []
    for x in word:
        position.append(i[0].lower().find(x))
    positions.append(position)
    #Se genera array con los términos de cada documento
    words.append(word)
    transformer = TfidfTransformer()
    #Se calcula TF-IDF de cada término de cada documento y se almacena en otro array
    tfidf = transformer.fit_transform(X)
    tfidfdoc.append(tfidf.toarray())

#Se genera dataframe donde cada línea representa un documento, se añade número de documento, el contenido del doc,
#los términos de ese documento y el valor TF-IDF de cada término del documento.
df = pd.DataFrame(foo, columns=['DocNumb', 'Document'])
df['Term Ind'] = positions
df['Terms'] = words
df['TF-IDF'] = tfidfdoc

print(df)