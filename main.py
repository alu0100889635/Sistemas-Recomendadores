from importlib.resources import contents
import sys

#Se lee fichero por línea de comandos
with open(sys.argv[1], 'r') as f:
    contents = [line.rstrip() for line in f]
print(contents)