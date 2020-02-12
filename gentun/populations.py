#!/usr/bin/env python
"""
Population class
"""

import itertools

from tqdm.auto import tqdm
from multiprocessing import Pool
from gentun.individuals import Individual


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

    @staticmethod
    def _get_fitness(individual: Individual):
        return individual, individual.get_fitness()

    def get_fittest(self):

        if self.individuals[-1].fitness is None:

            pool = Pool(self.n_workers)
            pbar = tqdm(total=len(self.individuals), desc='Evaluating individuals', leave=False)

            def update(*a):
                pbar.update()
                pbar.set_postfix_str(f"fitness={round(a[0][1], 4)}")

            jobs = [pool.apply_async(Population._get_fitness, args=(i,), callback=update)
                    for i in self.individuals]
            outputs = [p.get() for p in jobs]

            for i,res in enumerate(outputs):
                new_individual, _ = res
                self.individuals[i] = new_individual

            pool.close()
            pool.join()
            pbar.close()

            best = max if self.maximize else min
            return best(outputs, key=lambda x: x[1])[0]

        else:
            outputs = []
            for individual in self.individuals:
                fitness = individual.get_fitness()
                outputs.append((individual, fitness))

            best = max if self.maximize else min
            return best(outputs, key=lambda x: x[1])[0]

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
                 additional_parameters=None):
        if individual_list is None and genes_grid is None:
            raise ValueError("Either pass a list of individuals or a grid definition.")
        elif genes_grid is not None:
            genome = species(None, None).get_genome()  # Get species' genome
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
            maximize, additional_parameters
        )
