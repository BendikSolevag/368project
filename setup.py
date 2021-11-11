import sys
import subprocess

dependencies = ['transformers', 'inquirer', 'torch', 'sklearn', 'tqdm', 'simple_elmo', 'numpy', 'seqeval', 'conllu', 'tensorflow']
# implement pip as a subprocess:
for dependency in dependencies:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', dependency])
