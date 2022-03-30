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

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(prueba)
word = vectorizer.get_feature_names()
print(word)
print(X.toarray())

# transformer = TfidfTransformer()  
# print(transformer)

# tfidf = transformer.fit_transform(X)  
# print(tfidf.toarray())



#df = pd.DataFrame(foo, columns=['DocNumb', 'Document'])

#df = pd.DataFrame(word, columns=['Term'])
# df["Ind"] = " "
# df["Term"] = " "
# df["TF"] = " "
# df["TF-IDF"] = " "
# df["IDF"] = " "
#print(df)