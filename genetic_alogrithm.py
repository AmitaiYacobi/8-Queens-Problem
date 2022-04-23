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
        if score == 28: continue
        current += score
        if current > pick:
            return chromosome


def crossover(parent1, parent2):
    single_point = random.randint(0, 7)
    parent1 = list(parent1)
    parent2 = list(parent2)
    offspring1 = parent1[:single_point] + parent2[single_point:]
    offspring2 = parent2[:single_point] + parent1[single_point:]
    return tuple(offspring1), tuple(offspring2)


def mutation(offspring1, offspring2):
    for i in range(len(offspring1)):
        prop = random.randint(0, 7)
        offspring1 = list(offspring1)
        offspring2 = list(offspring2)
        if prop < 0.4:
            offspring1[i] = random.randint(0, 7)
            offspring2[i] = random.randint(0, 7)
    return tuple(offspring1), tuple(offspring2)


def create_population_scores_dict(population, scores):
    return {tuple(population[i]): scores[i] for i in range(len(scores))}


def run_algorithm(population_size=10):
    generation = 0
    population = generate_population(population_size)
    scores = population_fitness(population)
    population_scores_dict = create_population_scores_dict(population, scores)
    best_score = max(population_scores_dict.values())

    while best_score != 28:
        new_population = []
        for _ in range(int(len(population) / 2)):
            parent1 = selection(population_scores_dict)
            parent2 = selection(population_scores_dict)
            offspring1, offspring2 = crossover(parent1, parent2)
            offspring1, offspring2 = mutation(offspring1, offspring2)
            new_population.append(offspring1)
            new_population.append(offspring2)
            if len(new_population) == population_size:
                break

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
    population, num_of_generations = run_algorithm(population_size=1000)
    counter = 0
    for chromosome, score in population.items():
        if score == 28:
            counter += 1

    print("\n##################################")
    print(
        f"solution found after {num_of_generations} generations")
    print("##################################\n")
