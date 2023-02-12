import random
import copy
import numpy as np
import matplotlib.pyplot as plt


def initializePopulation(chromosome : list, popSize : int) -> list:
    '''Initialize population as a dict from a list'''

    population = []

    # Shuffle the chromosome to add individuals to population
    for _ in range(popSize):
        random.shuffle(chromosome)

        new = copy.deepcopy(chromosome)
        population.append(new)

    return population

def computeFitness(population : list, EDM : np.ndarray) -> list: # Funtion only for TSP
    '''Compute the fitness of each individual'''

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
    '''Tournament selection of tournament size k'''

    mating_pool = []

    for _ in range(len(population)):

        ind = random.randint(0, len(population)-1)
        bestFitness = fitness[ind]
        choice = copy.deepcopy(population[ind])
        
        for _ in range(k-1):

            ind = random.randint(0, len(population)-1)
            
            if fitness[ind] < bestFitness:
                bestFitness = fitness[ind]
                choice = copy.deepcopy(population[ind])
                
            mating_pool.append(choice)

    return mating_pool

def pmx(parent1, parent2) -> list:
    '''Partially Mapped Crossover'''

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
    '''Parents are selected randomly and the recombination is done according to a crossover rate'''

    offspring = []

    for _ in range(len(mating_pool)):

        # Choose parents randomly
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
    '''Swap mutation'''

    i = random.randint(0, len(chromosome)-1)
    j = random.randint(0, len(chromosome)-1)

    chromosome[i], chromosome[j] = chromosome[j], chromosome[i]

    return chromosome


def mutate(population : list, mutationRate : float) -> list:
    '''Mutation given a mutation rate'''

    offspring = []
     

    for individual in population:
        if random.random() < mutationRate:
            individual = swap(individual)

    offspring.append(individual)

    return offspring

def selectFittest(population, EDM : np.ndarray):
    '''Select the fittest menber of a population and returns it fitness'''

    fitness = computeFitness(population, EDM)
    bestFitness = np.array(fitness).min()
    fittest = population[fitness.index(bestFitness)]

    return fittest, bestFitness

def survivorSelection(offspring : list, population : list, EDM : np.ndarray) -> list:

    popFittest, popBest = selectFittest(population, EDM)
    _, offBest = selectFittest(offspring, EDM)

    if popBest < offBest:
        index = random.randint(0, len(offspring)-1)
        offspring[index] = popFittest
    
    population = copy.deepcopy(offspring)

    return population

class GeneticAlgorithm:
    def __init__(self, chromosome: list, popSize : int, crossoverRate : float, mutationRate : float, num_iter : int, show = False):
        self.chromosome = chromosome
        self.popSize = popSize
        self.crossoverRate = crossoverRate
        self.mutationRate = mutationRate
        self.num_iter = num_iter
        self.optimal_value = None 
        self.epsilon = None
        self.show = show

        # Performance Measures
        self.mean_best_fitness = None

    def showResults(self, x : list, y : list):
        '''Plot results'''

        plt.plot(x, y, label = 'Best fitness')
        plt.rc('text', usetex=True)
        plt.title(r'Progress for $\mu$: %s, $p_c$: %s, and $p_m$: %s'%(str(self.popSize), str(self.crossoverRate), str(self.mutationRate)))
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')

        best = self.optimal_value
        X = np.linspace(0, len(x))

        if best is not None:
            plt.plot(X, best*np.ones(len(X)), label = 'Optimal value')
        
        plt.legend()

        plt.savefig('progress.png', dpi=300)
        plt.show()

    def showStats(self):
        '''Show different Performance Measures'''
        print('The mean best fitness (MBF) is: %s'%(self.mean_best_fitness))

    def run(self, EDM):
        '''Run the Genetic Algorithm'''

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
            





