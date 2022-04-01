import sys
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#Se lee fichero por línea de comandos
with open(sys.argv[1], 'r') as f:
    contents = [line.rstrip() for line in f]

foo = list(map(lambda x: re.split(r".\s", x, 1), contents))

prueba = []

for i in foo:
    prueba.append(i[1])

test = []
for x in prueba:
    test.append([x])

words = []
tfidfdoc = []
for i in test:
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(i)
    word = vectorizer.get_feature_names_out()
    words.append(word)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    tfidfdoc.append(tfidf.toarray())


print(words)
print(tfidfdoc)

df = pd.DataFrame(foo, columns=['DocNumb', 'Document'])
df['Terms'] = words
df['TF-IDF'] = tfidfdoc

print(df)