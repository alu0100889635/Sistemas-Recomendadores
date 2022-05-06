import sys
import subprocess

# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'pandas'])

subprocess.check_call([sys.executable, '-m', 'pip', 'install', "-U", 
'scikit-learn'])

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'numpy'])