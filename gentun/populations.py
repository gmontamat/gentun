#!/usr/bin/env python
"""
Population class
"""

import itertools
from typing import List
import numpy as np

from tqdm.auto import tqdm
import multiprocessing as mp


def _tqdm_listener(queue: mp.Queue, total: int, desc: str, unit: str):
    """
      a process to start a tqdm and update it whenever a message on queue is received

    :param queue: a message queue
    :param total: total jobs to be done
    :param desc: progressbar description
    :param unit: progressbar unit
    :return:
    """

    progressbar = tqdm(total=total, desc=desc, unit=unit)
    for fitness in iter(queue.get, None):
        progressbar.update()
        progressbar.set_postfix_str(f"fitness={round(fitness, 4)}")

def split_list(a:List, n:int) -> List[List]:
    """
    Split the list a into n approximately equally long lists

    :param a: a list to split
    :param n: number of parts the list shall be split into
    :return: a list of lists
    """
    assert n>0

    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

class Population(object):
    """Group of individuals of the same species, that is,
    with the same genome. Can be initialized either with a
    list of individuals or a population size so that
    random individuals are created. The get_fittest method
    returns the strongest individual.
    """

    def __init__(self, species, x_train, y_train, individual_list=None, size=None,
                 crossover_rate=0.5, mutation_rate=0.015, maximize=True,
                 additional_parameters=None, n_workers=1):
        self.x_train = x_train
        self.y_train = y_train
        self.species = species
        self.n_workers = n_workers
        self.maximize = maximize
        if individual_list is None and size is None:
            raise ValueError("Either pass a list of individuals or a population size for a random population.")
        elif individual_list is None:
            if additional_parameters is None:
                additional_parameters = {}
            self.population_size = size
            self.individuals = [
                self.species(
                    self.x_train, self.y_train, crossover_rate=crossover_rate,
                    mutation_rate=mutation_rate, **additional_parameters
                )
                for _ in range(size)
            ]
        else:
            assert all([type(individual) is self.species for individual in individual_list])
            self.population_size = len(individual_list)
            self.individuals = individual_list

    def add_individual(self, individual):
        assert type(individual) is self.species
        self.individuals.append(individual)
        self.population_size += 1

    def get_species(self):
        return self.species

    def get_size(self):
        return self.population_size

    def _get_fittest_parallel(self):
        n = len(self.individuals)

        q = mp.Queue()
        proc = mp.Process(target=_tqdm_listener, args=(q, n, 'Evaluating individuals', 'ind.'))
        proc.start()

        mem_maps = self.individuals[0].prepare_data_sharing(n_workers=n,
                                                              n_colums=self.y_train.shape[0])

        indices = split_list(list(range(n)), self.n_workers)
        workers = []
        for i in range(self.n_workers):
            my_individuals = [self.individuals[j] for j in indices[i]]
            for individual in my_individuals:
                individual.clear_large_data()
            args = my_individuals, q, indices[i], *mem_maps
            workers.append(mp.Process(target=self.individuals[0].evaluate_fitness_and_return_results, args=args))

        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()

        q.put(None)  # finish the queue
        proc.join()  # finish the progressbarprocess

        self.individuals[0].update_individuals_from_remote_data(self.individuals, *mem_maps)

        best = np.argmax if self.maximize else np.argmin
        return self.individuals[best(mem_maps[0])]

    def _get_fittest_serial(self):
        if self.individuals[-1].fitness is None:
            t = tqdm(self.individuals, leave=False, desc="Evaluating individuals")
        else:
            t = self.individuals

        outputs = []
        for individual in t:
            fitness = individual.get_fitness()
            outputs.append((individual, fitness))
            if self.individuals[-1].fitness is None:
                t.set_postfix_str(f"fitness={round(fitness, 4)}")
        best = max if self.maximize else min
        return best(outputs, key=lambda x: x[1])[0]

    def get_fittest(self):
        if self.individuals[-1].get_fitness_status()==False and self.n_workers != 1:
            return self._get_fittest_parallel()
        else:
            return self._get_fittest_serial()

    def get_data(self):
        return self.x_train, self.y_train

    def get_fitness_criteria(self):
        return self.maximize

    def __getitem__(self, item):
        return self.individuals[item]


class GridPopulation(Population):
    """Population whose individuals are created based on a
     grid search approach instead of randomly. Can be
     initialized either with a list of individuals (in
     which case it behaves like a Population) or with a
     dictionary of genes and grid values pairs.
     """

    def __init__(self, species, x_train, y_train, individual_list=None, genes_grid=None,
                 crossover_rate=0.5, mutation_rate=0.015, maximize=True,
                 additional_parameters=None, n_workers=1):
        if individual_list is None and genes_grid is None:
            raise ValueError("Either pass a list of individuals or a grid definition.")
        elif genes_grid is not None:
            genome = species(None, None, **additional_parameters).get_genome()  # Get species' genome
            if not set(genes_grid.keys()).issubset(set(genome.keys())):
                raise ValueError("Some grid parameters do not belong to the species' genome")
            # Fill genes_grid with default parameters
            for gene, properties in genome.items():
                if gene not in genes_grid:
                    genes_grid[gene] = [properties[0]]  # Use default value
            individual_list = [
                species(
                    x_train, y_train, genes=genes, crossover_rate=crossover_rate,
                    mutation_rate=mutation_rate, **additional_parameters
                )
                for genes in (
                    dict(zip(genes_grid, x))
                    for x in itertools.product(*genes_grid.values())
                )
            ]
            print("Initializing a grid population. Size: {}".format(len(individual_list)))
        super(GridPopulation, self).__init__(
            species, x_train, y_train, individual_list, None, crossover_rate, mutation_rate,
            maximize, additional_parameters, n_workers=n_workers
        )
