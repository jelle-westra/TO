import sys
sys.path.append('../..')
from TO.utils import check_package_status

check_package_status(notebook=False)

from TO.core import Topology, ProblemInstance, run_experiment, OptimizationMethod
from TO.parameterizations.mmc import Capsules, MMCCenterpointsConfig, GuoBeam
from TO.problems.cantilever import create_horizontal_cantilever_problem

import argparse

def constructor_normalization_scale(n_components: int) -> ProblemInstance:
    topology = Topology(False, (100, 50), 1.0)
    parameterization = Capsules(topology, False, True, n_components, MMCCenterpointsConfig, GuoBeam(1000), 1000)

    rxmin = topology.domain_size_x/parameterization.n_components/(2 if parameterization.symmetry_x else 1)/2
    rymin = topology.domain_size_y/parameterization.n_components/(2 if parameterization.symmetry_y else 1)/2
    parameterization.normalization_scale[4] = min(rxmin, rymin)

    return create_horizontal_cantilever_problem(topology, parameterization)

def constructor_1components() : return constructor_normalization_scale(1)
def constructor_2components() : return constructor_normalization_scale(2) 
def constructor_5components() : return constructor_normalization_scale(5)
def constructor_10components() : return constructor_normalization_scale(10)
def constructor_20components() : return constructor_normalization_scale(20)

experiments = {
    10 : constructor_1components,
    20 : constructor_2components,
    50 : constructor_5components,
    100 : constructor_10components,
    200 : constructor_20components,
}

def parse_args() -> argparse.Namespace :
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dimension', type=int)
    parser.add_argument('-o', '--optimizer', type=str)
    parser.add_argument('-s', '--seed', type=int)
    return parser.parse_args()


if (__name__ == '__main__'):
    args = parse_args()
    method = {
        'CMAES': OptimizationMethod.CMAES, 
        'SMAC': OptimizationMethod.SMAC, 
        'HEBO': OptimizationMethod.HEBO, 
        'DE' : OptimizationMethod.DE,
        'BOTORCH' : None
    }[args.optimizer]

    budget = lambda parameterization : 100*(parameterization.dimension)
    problem: ProblemInstance = experiments[args.dimension]()

    name = f'{args.dimension}D_guo_{args.optimizer}'

    run_experiment(problem, budget(problem.parameterization), args.seed, name, method)