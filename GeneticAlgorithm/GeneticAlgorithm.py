"""Implementation of a Genetic Algortihm class and its functions for the TSP.
The modular implementation of the code allows it use in other applications.

Author:
    Francisco J. Palmero Moya @ UNED 
    12/08/2023
"""

import os
import copy
import random
import numpy as np
import matplotlib.pyplot as plt


def initializePopulation(chromosome : list, popSize : int) -> list:
    """Random initialisation of population.
    
    Parameters
    ----------
    chromosome : list
        Given individual genotype.
    popSize : int
        Desired number of individuals.

    Outputs
    -------
    population : list
        A multiset of genotypes.
    """

    population = []

    # Shuffle the chromosome to add individuals to population
    for _ in range(popSize):
        random.shuffle(chromosome)

        new = copy.deepcopy(chromosome)
        population.append(new)

    return population

def computeFitness(population : list, EDM : np.ndarray) -> list: # Funtion only for TSP
    """Compute the fitness of each individual.
    
    Parameters
    ----------
    popolutaion : list
        A multiset of genotypes.
    EDM : array
        Euclidean Distance Matrix between cities.

    Outputs
    -------
    fitness : list
        Fitness of each individual as a list.
    """

    fitness = []

    # Compute fitness for each individual in population
    for ind in range(len(population)):

        # Copy of the chromosome
        chromosome = copy.deepcopy(population[ind])

        # Initialize the sum == 'Total path distance'
        sum = 0
        for genenr in range(len(chromosome)-1):
            i = chromosome[genenr].number
            j = chromosome[genenr+1].number

            sum += EDM[i, j]

        # Add distance to go back to the origin
        i = chromosome[-1].number
        j = chromosome[0].number
        sum += EDM[i, j]

        # Save the fitness
        fitness.append(sum)
    
    return fitness

def selectParents(population : list, fitness : list, k : int = 2) -> list:
    """Tournament selection of tournament size k. The number of parents selected
    is equal to the population size.
    
    Parameters
    ----------
    population : list
        A multiset of genotypes.
    fitness : list
        Fitness of each individual as a list.
    k : int
        Tournament size.
    
    Outputs
    -------
    mating_pool : list
        Parents of the next generation.
    """

    mating_pool = []

    # Select popSize parents
    for _ in range(len(population)):

        # Pick k individuals randomly with replacement.
        # The comparison starts setting one candidate as best individual.
        ind = random.randint(0, len(population)-1)
        bestFitness = fitness[ind]
        choice = copy.deepcopy(population[ind])
        
        for _ in range(k-1):

            ind = random.randint(0, len(population)-1)
            
            # If the individial is better, update the candidate 
            if fitness[ind] < bestFitness:
                bestFitness = fitness[ind]
                choice = copy.deepcopy(population[ind])
            
            # Include the best individual of the tournament.
            mating_pool.append(choice)

    return mating_pool

def pmx(parent1 : list, parent2 : list) -> list:
    """Crossover as Partially Mapped Crossover.
    
    Parameters
    ----------
    parent1 : list
        Individual genotype 1 as permutation representation.
    parent2 : list
        Individual genotype 1 as permutation representation.
    
    Outputs
    -------
    offspring : list
        Offspring genotype after crossover.
    """

    # Choose two random cutting points
    cutting_point1 = random.randint(0, len(parent1) - 1)
    cutting_point2 = random.randint(0, len(parent1) - 1)
    
    # Make sure cutting_point1 is smaller than cutting_point2
    if cutting_point1 > cutting_point2:
        cutting_point1, cutting_point2 = cutting_point2, cutting_point1
    
    
    # The segment that will be exchanged between parents
    segment = parent1[cutting_point1:cutting_point2+1]
    
    # Initialize the mapping
    mapping = {}
    for i in range(cutting_point1, cutting_point2+1):
        mapping[parent2[i]] = parent1[i]
        
    # Initialize the offspring
    offspring = [-1] * len(parent1)
    
    # Fill the offspring with the segment
    offspring[cutting_point1:cutting_point2+1] = segment
    
    # Fill the rest of the offspring using the mapping
    for i in range(len(parent1)):
        if i >= cutting_point1 and i <= cutting_point2:
            continue
        offspring[i] = parent2[i]
        while offspring[i] in segment:
            offspring[i] = mapping[offspring[i]]
    
    return offspring

def recombination(mating_pool : list, crossoverRate : float) -> list:
    """Parents are selected randomly and the recombination is done according to a crossover rate.
    
    Parameters
    ----------
    mating_pool : list
        Selected parents from the population.
    crossoverRate : float
        Probability of crossover
    
    Outputs
    -------
    offspring : list
        Offspring genotype after recombination.
    """

    offspring = []

    # Create offspring after recombination
    for _ in range(len(mating_pool)):

        # Choose parents randomly with replacement.
        parent1 = copy.deepcopy(mating_pool[random.randint(0, len(mating_pool)-1)])
        parent2 = copy.deepcopy(mating_pool[random.randint(0, len(mating_pool)-1)])

        # PMX with probability = crossover rate
        if random.random() < crossoverRate:
            child = pmx(parent1, parent2)
        else:
            child = copy.deepcopy(parent1)

        offspring.append(child)

    return offspring

def swap(chromosome : list) -> list:
    """Swap mutation,
    
    Parameters
    ----------
    chromosome : list
        Individual genotype.
    
    Outputs
    -------
    chromosome : list
        Individual genotype after swap mutation
    """

    # Select two genes at random in the chromosome
    i = random.randint(0, len(chromosome)-1)
    j = random.randint(0, len(chromosome)-1)

    # Swap allele values
    chromosome[i], chromosome[j] = chromosome[j], chromosome[i]

    return chromosome


def mutate(population : list, mutationRate : float) -> list:
    """Mutation given a mutation rate
    
    Parameters
    ----------
    population : list
        A multiset of genotypes.
    mutationRate : float
        Probability of crossover
    
    Outputs
    -------
    offspring : list
        Population after mutation
    """

    offspring = []
     
    # Mutate population with probability = mutationRate
    for individual in population:
        if random.random() < mutationRate:
            individual = swap(individual)

    offspring.append(individual)

    return offspring

def selectFittest(population : list, EDM : np.ndarray):
    """Select the fittest menber of a population and returns it fitness.
    
    Parameters
    ----------
    population : list
        A multiset of genotypes.
    EDM : array
        Euclidean Distance Matrix between cities.
    
    Outputs
    -------
    fittest : list
        The fittest individual in the population.
    bestFitness : list
        The fitness of the fittest individual in the population.
    """

    fitness = computeFitness(population, EDM)
    bestFitness = np.array(fitness).min()
    fittest = population[fitness.index(bestFitness)]

    return fittest, bestFitness

def survivorSelection(offspring : list, population : list, EDM : np.ndarray) -> list:
    """Survivor selection mechanism as a generational model. Elitism is applied.
    
    Parameters
    ----------
    offspring : list
        Candidate individuals for the new generation.
    population : list
        Old generation individuals.
    EDM : array
        Euclidean Distance Matrix between cities.

    Outputs
    -------
    population : list
        Updated population after survivor selection.
    """

    # Select fittest individual of population
    popFittest, popBest = selectFittest(population, EDM)

    # Evaluate offspring fitness
    _, offBest = selectFittest(offspring, EDM)

    # Apply elitism
    if popBest < offBest:
        # Select individial for replacement at random
        index = random.randint(0, len(offspring)-1)
        offspring[index] = popFittest
    
    population = copy.deepcopy(offspring)

    return population

class GeneticAlgorithm:
    def __init__(self, chromosome: list, popSize : int, crossoverRate : float, mutationRate : float, num_iter : int, show = False):
        """Genetic Algortihm.
        
        Parameters
        ----------
        chromosome : list
            Given individual genotype.
        popSize : int
            Desired number of individuals.
        crossoverRate : float
            Probability of crossover.
        mutationRate : float
            Probability of mutation.
        num_iter : int
            Maximun number of iterations.
        show : bool
            Show results after running.
        """
        self.chromosome = chromosome
        self.popSize = popSize
        self.crossoverRate = crossoverRate
        self.mutationRate = mutationRate
        self.num_iter = num_iter
        self.show = show

        # Set optimal value (if it is known a priori) and error margin
        self.optimal_value = None 
        self.epsilon = None

        # Performance Measures
        self.mean_best_fitness = None

    def showResults(self, x : list, y : list, filename : str = None):
        """Plot progress history."""

        plt.plot(x, y, label = 'Best fitness')
        plt.rc('text', usetex=True)
        plt.title(r'Progress history for $\mu$: %s, $p_c$: %s, and $p_m$: %s'%(str(self.popSize), str(self.crossoverRate), str(self.mutationRate)))
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')

        best = self.optimal_value
        X = np.linspace(0, len(x))

        if best is not None:
            plt.plot(X, best*np.ones(len(X)), label = 'Optimal value')
        
        plt.legend()

        if filename is not None:
            plt.savefig(os.path.join('figures', filename), dpi=300)
        plt.show()

    def showStats(self):
        """Show different Performance Measures"""
        print('The mean best fitness (MBF) is: %s'%(self.mean_best_fitness))

    def run(self, EDM):
        """Run the Genetic Algorithm"""

        # Variables to plot
        y = [] # Best fitness
        x = [] # Current interation

        # Initialize the population
        population = initializePopulation(self.chromosome, self.popSize)

        for iter in range(self.num_iter):
            # Population evaluation
            fitness = computeFitness(population, EDM)
            currentBest = np.array(fitness).min()

            # Save progress to plot
            y.append(currentBest)
            x.append(iter)

            # Termination condition
            if self.optimal_value is not None:
                if currentBest < self.optimal_value + self.epsilon:
                    break

            # Parents selection
            mating_pool = selectParents(population, fitness)

            # Create offspring: recombination, mutation
            offspring = recombination(mating_pool, self.crossoverRate)
            offspring = mutate(offspring, self.mutationRate)

            # Survivor selection
            population = survivorSelection(offspring, population, EDM)
        
        fittest, bestFitness = selectFittest(population, EDM)
        self.mean_best_fitness = np.array(y).mean()

        if self.show:
            self.showResults(x, y)

        
        return fittest, bestFitness
            





