import numpy as np

import os
import subprocess
from glob import glob
from IPython.display import display, Markdown
from typing import List

def check_package_status(notebook: bool=True):
    path = os.path.dirname(os.path.abspath(__file__))
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)

    file_changes = subprocess.run(['git', 'diff', '--name-only', path], capture_output=True, text=True).stdout
    untracked_files = subprocess.run(['git', 'ls-files', '--others', '--exclude-standard', path], capture_output=True, text=True).stdout
    
    version = commit.stdout.strip()
    if (notebook):
        if (file_changes or untracked_files):
            display(Markdown(
                '<div class="alert alert-block alert-danger" style="font-size: 14px; border-radius: 10px">'
                f'<h4>[NO GO] TO @ {version}</h4>'
                '<pre>  ⚠️ Changes found in the ./TO package, first commit changes before starting experiments.</pre>'
                '</div>'
            ))
        else: 
            display(Markdown(
                '<div class="alert alert-block alert-success" style="font-size: 14px; border-radius: 10px">'
                f'<h4>[GO] TO @ {version}</h4>'
                '<pre>  ✅ No changes found in the ./TO packge.</pre>'
                '</div>'
            ))
    else:
        if (file_changes or untracked_files) :
            raise AssertionError('Changes found in the ./TO package, first commit changes before starting experiments.')
        if (version_files := glob('*.version')) :
            assert (len(version_files) == 1), 'Can not continue w/ multiple version files; only a sinlge version can be attributed to an experiment.'
            if ((version_prev := version_files[0].split('.')[0]) != version) : 
                raise AssertionError(f'Current version ({version}) is different from previously run instances of the experiment ({version_prev}).')
    with open(f'{version}.version', 'w') as _ : ...
    return True          


def read_evals(name: str, seed: int, run: int=None) -> List[str] :
    path = os.path.join(os.path.abspath(''), f'results/{name}/{seed}/evals.dat' if not(run) else f'results/{name}/{seed}-{run}/evals.dat')
    with open(path, 'r') as handle : return handle.readlines()

def get_fitness_values(name: str, seed: int, run: int=None) -> np.ndarray :
    lines = read_evals(name, seed, run)
    f = np.array([float(line.split()[1]) for line in lines])
    illegal = np.array([float(line.split()[2]) > 0 for line in lines])
    f[illegal] = float('inf')
    return f

def get_best_config(name:str, seed: int, run: int=None) -> np.ndarray : 
    lines = read_evals(name, seed, run)
    line_best = min(lines, key=lambda line : float(line.split()[1])) 
    return np.array([float(xi) for xi in line_best.split()[4:]])

def get_config(evaluation: int, name:str, seed: int, run: int=None) -> np.ndarray :
    lines = read_evals(name, seed, run)
    return np.array([float(xi) for xi in lines[evaluation+1].split()[4:]])