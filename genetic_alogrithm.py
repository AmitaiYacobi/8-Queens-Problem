import random
import numpy as np

BEST_SCORE = 28

def get_key(dictionary, val):
    for key, value in dictionary.items():
        if val == value:
            return key
    return "key doesn't exist"


def generate_chromosome(chromosome_size):
    return [random.randint(0, 7) for _ in range(chromosome_size)]


def generate_population(population_size=100, chromosome_size=8):
    population = []
    for _ in range(population_size):
        population.append(generate_chromosome(chromosome_size))
    return population

# the max number of clashes in chromosome is 28
# (for example, the chromosome [1,1,1,1,1,1,1,1] has 8choose2 clashes which is 28)
# so fitness score will be 28 minus number of clashes in a chromosome
# so it means that the best score is 28
def population_fitness(population):
    scores = []
    for chromosome in population:
        clashes = 0
        row_clashes = abs(len(chromosome) - len(np.unique(chromosome)))
        clashes += row_clashes
        for i in range(len(chromosome)):
            for j in range(i + 1, len(chromosome)):
                if abs(i-j) == abs(chromosome[i] - chromosome[j]):
                    clashes += 1

        scores.append(28 - clashes)
    return scores

def chromosome_fitness(chromosome):
    clashes = 0
    row_clashes = abs(len(chromosome) - len(np.unique(chromosome)))
    clashes += row_clashes
    for i in range(len(chromosome)):
        for j in range(i + 1, len(chromosome)):
            if abs(i-j) == abs(chromosome[i] - chromosome[j]):
                clashes += 1
    return 28 - clashes

def selection(population_scores_dict):
    sum_of_scores = sum(population_scores_dict.values())
    pick = random.uniform(0, sum_of_scores)
    current = 0
    for chromosome, score in population_scores_dict.items():
        current += score
        if current > pick:
            return chromosome

def singlepoint_crossover(parent1, parent2, rate="random"):
    if rate == "random":
        single_point = random.randint(0, len(parent1))
    else:
        single_point = rate / len(parent1)

    parent1 = list(parent1)
    parent2 = list(parent2)
    offspring1 = parent1[:single_point] + parent2[single_point:]
    offspring2 = parent2[:single_point] + parent1[single_point:]
    return tuple(offspring1), tuple(offspring2)

def crossover(parent1, parent2, crossove_type=singlepoint_crossover, rate="random"):
    """

    :param parent1: first parent (tuple)
    :param parent2: second parent (tuple)
    :param crossove_type: function for crossover. get arguments: parent1, parent2, crossover rate
    :param rate: rate for crossover
    :return: two children
    """
    return crossove_type(parent1, parent2, rate=rate)


def mutation(offspring1, offspring2, rate=0.4):
    offspring1 = list(offspring1)
    offspring2 = list(offspring2)
    for i in range(len(offspring1)):
        if random.random() < rate: # to do mutation
            offspring1[i] = random.randint(0, 7)
            offspring2[i] = random.randint(0, 7)
    return tuple(offspring1), tuple(offspring2)


def create_population_scores_dict(population, scores):
    return {tuple(population[i]): scores[i] for i in range(len(scores))}


def elitism(popul_scores_dict, p=0.1):
    """

    :param popul_scores_dict: key: chromosom, value: score
    :param p: float (percentage) how many besh chromosoms to pass to the next generation.
    :return: list of best chromosoms
    """
    n_best = int(p * len(popul_scores_dict))
    sorted_popul = sorted(popul_scores_dict.items(), key=lambda item: item[1], reverse=True)
    best = sorted_popul[:n_best]
    best_rep = [s[0] for s in best] # save the representation of each solution from the best ones
    return best_rep


def num_of_solutions(population_scores_dict):
    solutions = 0
    for c, v in population_scores_dict.items():
        if v == 28:
            solutions += 1
    return solutions


def run_algorithm(population_size, crossover_type, crossover_rate, mutation_rate, p_elitism):
    generation = 0
    population = generate_population(population_size)
    scores = population_fitness(population)
    population_scores_dict = create_population_scores_dict(population, scores)
    # best_score = max(population_scores_dict.values())

    while num_of_solutions(population_scores_dict) < N_SOLUTIONS: # best_score != 28:
        new_population = []
        new_population.extend(elitism(population_scores_dict, p=p_elitism))
        remain = len(population) - len(new_population)
        for _ in range(remain // 2): # run all population except the elitism we pass
            parent1 = selection(population_scores_dict)
            parent2 = selection(population_scores_dict)
            offspring1, offspring2 = crossover(parent1, parent2, crossove_type=crossover_type, rate=crossover_rate)
            offspring1, offspring2 = mutation(offspring1, offspring2, rate=mutation_rate)
            new_population.append(offspring1)
            new_population.append(offspring2)
            # if len(new_population) == population_size:
            #     break

        population = new_population
        new_scores = population_fitness(new_population)
        population_scores_dict = create_population_scores_dict(
            new_population, new_scores)
        best_score = max(population_scores_dict.values())
        best_chromosome = get_key(population_scores_dict, best_score)
        print(f"chromosome: {best_chromosome} score: {best_score}")
        generation += 1

    return population_scores_dict, generation


if __name__ == "__main__":
    config = {
        "population_size" : 300,
        "crossover_type": singlepoint_crossover,
        "crossover_rate": "random",
        "mutation_rate": 0.2,
        "p_elitism": 0.2
    }
    N_SOLUTIONS = 10
    solutions = []
    population, num_of_generations = run_algorithm(**config)
    print("finish one running")
    for chromosome, score in population.items():
        if score == 28:
            solutions.append(chromosome)

    print(f"sum of generations: {num_of_generations}")
    print("solutoins:")
    print(solutions)

    # print("\n##################################")
    # print(
    #     f"solution found after {num_of_generations} generations")
    # print("##################################\n")
