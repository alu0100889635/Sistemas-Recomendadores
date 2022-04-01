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

df = pd.DataFrame(foo, columns=['DocNumb', 'Document'])
df['Term Ind'] = positions
df['Terms'] = words
df['TF-IDF'] = tfidfdoc

print(df)