import sys

#Se lee fichero por línea de comandos
with open(sys.argv[1], 'r') as f:
    contents = f.read()
print(contents)