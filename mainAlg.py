from builtins import type
from deap import base, tools, algorithms, benchmarks, creator
import numpy as np
from operator import itemgetter
import random, math
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from random import randint
from collections import OrderedDict


def binaryInit(popSize, stSize):
    """

    :param popSize: population size
    :param stSize: string size
    :return: a binary string
    """
    pop = []

    for _ in range(popSize):
        x = ''  # this needs to be reset every iteration
        for _ in range(int(stSize)):
            x += ''.join(random.choice(['0', '1']))
        pop.append(x)
    return pop


mutate_mapping = {'0': '1', '1': '0'}
SIZE = 0


def bitWiseNoise(individual, p, q):
    """

    :param p: With prob p -> f(x) and with prob 1-p  -> f(x')
    :param q: Prob of flipping each bit (p_mut)
    :return: Mutated individual or original with respect to p
    """
    SIZE = len(individual)
    modified = [_ for _ in individual]
    for i in range(SIZE):
        # Flipping bits with prob q
        if random.random - q < 0:
            modified[i] = mutate_mapping[modified[i]]

    if random.random() - p < 0:
        return modified
    else:
        return individual


def bitWiseNoiseOneMax(individual, p, q):
    """

    :param p: Probability of returning the noisy value
    :param q: Probability of flipping each bit
    :return: Noisy fitness value of One-Max which is number of Ones
    """
    sum = 0
    if_noise = 0
    for idx, i in list(enumerate(individual, 0)):
        sum += int(i)
    if random.random() - p < 0:
        new_val = bitWiseNoise(individual, p, q)
        for j in new_val:
            if_noise += int(j)
        return if_noise
    else:
        return sum


def oneBitNoise(individual):
    '''
    With probability of (1-p_n) original one, with probability of p_n  modified
    only one bit gets flipped
    :param p_n: noise prob
    :param individual: binary array
    :return:    modified or non modified individual
    '''
    SIZE = len(individual)
    prob = random.random()
    modified = [_ for _ in individual]
    ind = np.random.choice(SIZE)
    modified[ind] = mutate_mapping[modified[ind]]  # selecting an index uniformly and flip

    return modified


def oneMax(individual):
    """

    :param individual: Individual to be evaluated
    :return: Fitness value of the individual(num of 1s)
    """
    sum = 0

    for idx, i in list(enumerate(individual, 0)):
        sum += int(i)
    return sum


def asymmetricOneBit(individual):
    """
    returned individual is "modified" just for the naming sake
    :param individual: individual whose bit is going to be flipped
    :return: modified individual
    """
    # probability of flipping a zero bit is (1/size of zeros) * 1/2
    zero_size = 0
    SIZE = len(individual)
    zero_idx = []
    one_idx = []
    idx_bins = []

    modified = list(individual)
    for i in range(SIZE):
        if individual[i] == '0':
            zero_idx.append(i)
            idx_bins.append(i)
            zero_size += 1
        else:
            one_idx.append(i)
            idx_bins.append(i)
    prob = random.random()
    # string length - zero_size gives the size of ones.
    one_size = len(modified) - zero_size
    zero_one_prob = random.random()
    if zero_size == 0 or zero_size == SIZE:
        # select an index uniformly and flip
        # idx is going to be the array of indexes which will be sel
        my_idx = random.choice(idx_bins)
        modified[my_idx] = ''.join(mutate_mapping[individual[my_idx]])
    # with prob of 1/2 * 1/zero_size zero flip gets flipped
    elif zero_one_prob > 1 / 2:
        if prob > (1 / 2 * (1 / zero_size)):
            sel_z_idx = np.random.choice(zero_idx)
            modified[sel_z_idx] = mutate_mapping[individual[sel_z_idx]]

    else:
        if prob > (1 / 2 * (1 / one_size)):
            # select a ones index and flip
            sel_one_idx = np.random.choice(one_idx)
            modified[sel_one_idx] = mutate_mapping[modified[sel_one_idx]]

    return modified


def randomLocalSearch():
    """
    Implementation of RLS, this will not be used.
    :return: runtime
    """
    population = binaryInit(100, 10)
    x = random.choice(population)
    bin_size = len(x)
    all_idx = []
    generation = 1

    fitness_iters = 0
    for idxs, bit_arr in list(enumerate(x, 0)):
        all_idx.append(idxs)

    while generation != 500:
        new_ind = [_ for _ in x]
        flip_idx = random.choice(all_idx)
        new_ind[flip_idx] = mutate_mapping[x[flip_idx]]
        generation += 1
        fitness_value = oneMax(new_ind)
        fitness_iters += 1
        if fitness_value == bin_size:
            return fitness_iters
        if fitness_value >= oneMax(x):
            x = new_ind
    return fitness_iters


def asymmetricNoiseOneMax(individual, p_n):
    """

    :param individual: individual to be evaluated
    :param p_n: noise probability
    :return: noisy fitness value with prob p_n
    """
    # res = 0
    # if_noise = 0
    # new_val = ''
    # for idx, i in list(enumerate(individual, 0)):
    #    res += int(i)
    # if random.random() - p_n < 0:
    #    for i in (asymmetricOneBit(individual)):
    #        new_val += i
    #    for j in new_val:
    #        if_noise += int(j)
    #    return if_noise
    # else:
    #   return res
    if random.random() - p_n < 0:
        return oneMax(asymmetricOneBit(individual))
    else:
        return oneMax(individual)


def oneBitNoiseOneMax(individual, p_n):
    """

    :param individual: Individual to be evaluated
    :param p_n: Noise probability
    :return: Noisy fitness value
    """
    # sum = 0
    # if_noise = 0
    # for idx, i in list(enumerate(individual, 0)):
    #    sum += int(i)
    # if random.random() - p_n < 0:
    #    new_val = oneBitNoise(individual, p_n)
    #    for j in new_val:
    #        if_noise += int(j)
    #    return if_noise
    # else:
    #    return sum

    if random.random() - p_n < 0:
        return oneMax(oneBitNoise(individual))
    else:
        return oneMax(individual)


def selectNoiseParam(gam1, gam2):
    noise_param = random.choice(gam1, gam2)
    return noise_param


def additiveGaussianOneMax(individual):
    # According to Qian et al.2018,  gamma1 - gamma2 < 2n
    # sigma <= 1 (Qian et al. 2018) for poly runtime
    mu, sigma = 0, 1
    gam1, gam2 = np.random.normal(mu, sigma, 2)
    noise = random.choice((gam1, gam2))
    return sum(individual) + noise


def randomSelection(individuals, tourn_size):
    return [random.choice(individuals) for i in range(tourn_size)]


def tournSelection(individuals, tourn_size, k, fitness_func):
    best_inds = []
    for i in range(k):
        participants = randomSelection(individuals, tourn_size)
        best_inds.append(max(participants), key=fitness_func(participants))
    return best_inds


def muPlusLambdaEA(mu, lambda_, bin_size, p_mut, n_type, p_noise):
    """
    (Mu + Lambda)-EA implementation on OneMax problem.
    :param mu: Population size
    :param lambda_: Offspring size
    :param bin_size: string size
    :param p_mut: Mutation probability
    :param n_type: Noise type if this is 'one', one Bit Noise will be used, asymmetric one Bit if 'asym' and no noise in other cases
    :param p_noise: Noise strength
    :return: Fitness iterations until optimal value
    """
    population = binaryInit(mu, bin_size)
    new_population = population
    fitness_value = 0

    generation = 1
    fitness_iters = 0

    ind_fit_vals = {}
    for ind in population:
        ind_fit_vals[ind] = oneMax(ind)
        fitness_iters += 1

    while generation != 500:
        for i in range(lambda_):
            x = random.choice(new_population)
            try_element = [_ for _ in x]
            for bit in range(len(x)):
                if random.random() - p_mut < 0:
                    try_element[bit] = mutate_mapping[x[bit]]

            try_element = ''.join(try_element)
            if n_type == 'one':
                fitness_value = oneBitNoiseOneMax(try_element, p_noise)
            elif n_type == 'asym':
                fitness_value = asymmetricNoiseOneMax(try_element, p_noise)
            else:
                fitness_value = oneMax(try_element)
            fitness_iters += 1
            ind_fit_vals[try_element] = fitness_value
            # fitness_value = asymmetricNoiseOneMax(try_element, p_noise)


            if fitness_value == bin_size:
                return fitness_iters

        new_population = list(OrderedDict(sorted(ind_fit_vals.items(), key=itemgetter(1), reverse=True)).keys())
        new_population = new_population[:mu]
        generation += 1
    return fitness_iters


def onePLambda_EA(lambda_, bin_size, p_mut, n_type, p_noise):
    """
    Select a random individiaul from 100 randomly generated binaries of "bin_size" and run the algorithm
    :param lambda_:  offspring size
    :param bin_size: binary size
    :param p_mut: mutation probability
    :param n_type: Noise type if this is 'one', one Bit Noise will be used, asymmetric one Bit if 'asym' and no noise in other cases
    :param p_noise: Noise strength
    :return: fitness iterations until optimal solution
    """
    init_pop_size = 100
    population = binaryInit(init_pop_size, bin_size)
    x = random.choice(population)
    init_var = x  # THIS WILL NOT CHANGE

    # fitness_values = []
    fitness_value = 0
    # unnecessary_individuals = []
    fitness_each_gen = {}

    # sum = np.inf
    generations = 1
    fitness_iters = 0
    # so while we do not have optimum solution from our x we create lambda mutated solutions.
    # We pick the fittest then compare with original.
    while generations != 500:
        off_fit_vals = {}
        orig_fit_val = oneMax(x)
        fitness_values = []
        # iterating over offspring
        for i in range(lambda_):

            offspring = [_ for _ in x]
            # new_ind = []
            # iterating over indexes
            for idx in range(bin_size):
                if p_mut - random.random() < 0:
                    offspring[idx] = mutate_mapping[x[idx]]

            offspring = ''.join(offspring)
            # new_ind.append(offspring)
            if n_type == 'one':
                fitness_value = oneBitNoiseOneMax(offspring, p_noise)
            elif n_type == 'asym':
                fitness_value = asymmetricNoiseOneMax(offspring, p_noise)
            else:
                fitness_value = oneMax(offspring)
            fitness_iters += 1
            if (offspring == '1' * bin_size):
                return fitness_iters
            fitness_values.append(fitness_value)

            off_fit_vals[fitness_value] = offspring

            # oneB_fitness_values.append(oneBitNoiseOneMax(x, 0.1))
        max_fitness_value = np.max(fitness_values)

        if (max_fitness_value >= orig_fit_val):
            x = off_fit_vals.get(max_fitness_value)
        # if(oneb_max_fitness_value) > fitness_value):

        fitness_each_gen[
            generations] = max_fitness_value  # this exists just to show the change of the fitness in every generation.

        generations += 1

    return fitness_iters


def tournSelEA(str_size, p_mut, n_type, p_noise):
    """
    Lambda(population size will be found according to string length.
    Binary tournament selection will be used
    for i to lambda:
        - sample two parents randomly selected
        - tournament selection
        - flip each bit with prob p_mut
    :param str_size: bitsring length
    :param p_mut: Mutation probability
    :param n_type: Noise type if this is 'one', one Bit Noise will be used, asymmetric one Bit if 'asym' and no noise in other cases
    :param p_noise: Noise strength
    :return: Number of fitness iterations until optimal solution is reached
    """
    mu = int(np.ceil(13 * np.log(str_size)))  # according to (Dang&Lehre 2014)-> lambda

    lambda_ = mu // str_size

    population = binaryInit(mu, str_size)

    gen = 0
    fitness_iters = 0
    while True:
        new_population = []
        for i in range(mu):
            new_individual = []
            # Randomly select among the population
            parent_one, parent_two = randomSelection(population, 2)
            if n_type == 'one':
                fit_val_one = oneBitNoiseOneMax(parent_one, p_noise)
                fit_val_two = oneBitNoiseOneMax(parent_two, p_noise)
            if n_type == 'asym':
                fit_val_one = asymmetricNoiseOneMax(parent_one, p_noise)
                fit_val_two = asymmetricNoiseOneMax(parent_two, p_noise)
            else:
                fit_val_one = oneMax(parent_one)
                fit_val_two = oneMax(parent_two)

            fitness_iters += 2

            # Evaluate fitness each time check if it is the optimal solution or not

            # Now do the selection

            if fit_val_one > fit_val_two:
                new_individual = parent_one
            if fit_val_two > fit_val_one:
                new_individual = parent_two
            else:
                new_individual = random.choice([parent_one, parent_two])
            new_individual = [_ for _ in new_individual]
            # Mutation
            for i in range(str_size):
                if random.random() - p_mut < 0:
                    new_individual[i] = mutate_mapping[new_individual[i]]
            new_individual = ''.join(new_individual)
            new_population.append(new_individual)
            if new_individual == ('1' * str_size):
                return fitness_iters
        population = new_population


trying = ['0', '1', '1', '1', '1']
a = 0
idx = list(enumerate(trying, 0))
asym_results = []
oneB_results = []
orig_results = []
muPOne_results = []
asym_means = []
oneB_means = []
orig_means = []
muPOne_means = []


def findMutationProbability(mu, lambda_, str_size):
    """

    :param mu: Population size
    :param lambda_: Offspring size
    :return: mutation probability "x/n"
    """
    for i in np.arange(0.1, 2, 0.1):
        pw = np.exp(i)
        if pw < (mu / lambda_):
            p_mut = i / str_size
            return p_mut
            break


# print(type(random.choice(idx[0])))
# print(idx)
def muCommaLambdaEA(str_size, n_type, p_noise):
    """
    In this algorithm, individuals are sorted according to their fitness levels(canonical partition)
    After sorting best lambda are selected.
    In this algorithm individuals will be aligned with their fitness levels
     so further into computation if an individual is built again fitness value can be selected
    :param population: Initial population
    :param mu: Population size
    :param lambda_: Selected individual size
    Mutation probability will be found with findMutationProbability function.
    In notation Dr Lehre uses lambda as population size and Mu as the offspring size. I've used it vice versa.
    :return: fitness iterations
    """
    # Map structure will be x[individual] = fitness_value
    # Initialisation of the population
    lambda_ = int(np.ceil(100 * np.log(str_size)))
    mu = lambda_ // 10
    population = binaryInit(lambda_, str_size)
    p_mut = findMutationProbability(lambda_, mu, str_size)
    ind_fitness_pairs = {}
    gen = 0
    fitness_iters = 0
    while gen != 500:
        new_population = []
        ind_fitness_pairs = {}
        for x in population:

            ind = x

            if n_type == 'one':
                fitness_value = oneBitNoiseOneMax(ind, p_noise)
            elif n_type == 'asym':
                fitness_value = asymmetricNoiseOneMax(ind, p_noise)
            else:
                fitness_value = oneMax(ind)
            if fitness_value == str_size:
                return fitness_iters
            fitness_iters += 1
            ind_fitness_pairs[ind] = fitness_value
        sorted(ind_fitness_pairs.items(), key=lambda x: x[1], reverse=True)  # sort by value in dec order
        # Selection(lambda individuals will be selected)
        # Sort P_t such that f(P_t(1)) >= f(P_t(2))
        # For population size, select one individual from the sorted part and mutate it DO NOT CHANGE JUST COPY
        new_population = list(ind_fitness_pairs.keys())  # we are not interested in fitness values in this part
        select_part = new_population[:mu]  # selecting the first lambda individuals
        new_population = []
        # Mutation
        for j in range(lambda_):

            ind1 = random.choice(select_part)
            if ind1 == '1' * str_size:
                return fitness_iters
            ind2 = random.choice(select_part)
            if ind2 == '1' * str_size:
                return fitness_iters
            ind1 = [_ for _ in ind1]
            ind2 = [_ for _ in ind2]
            ind1, ind2 = crossover(ind1, ind2)
            ind1 = random.choice([ind1, ind2])
            for i in range(str_size):

                if random.random() - p_mut < 0:
                    ind1[i] = mutate_mapping[ind1[i]]
            mutated_individual = ''.join(ind1)
            new_population.append(mutated_individual)
            if mutated_individual == '1' * str_size:
                return fitness_iters
                # After mutation the real selection part which is selecting best individuals
                # selecting the first mu individuals
        population = new_population
        # Map structure will be x[individual] = fitness_value
        # Initialisation of the population
        return fitness_iters


def crossover(ind1, ind2):
    # Change a bit of a random index select uniformly from one of the individuals
    x = ind1
    y = ind2
    for i in range(len(ind1)):
        idxs = [ind1[i], ind2[i]]
        f_idx = random.choice(idxs)
        idxs.remove(f_idx)
        s_idx = idxs[0]
        x[i] = f_idx
        y[i] = s_idx
    return x, y
