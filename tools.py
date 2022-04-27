import matplotlib.pyplot as plt


def get_key(dictionary, val):
    for key, value in dictionary.items():
        if val == value:
            return key
    return "key doesn't exist"

def create_population_scores_dict(population, scores):
    return {tuple(population[i]): scores[i] for i in range(len(scores))}

def num_of_solutions(population_scores_dict):
    solutions = 0
    for c, v in population_scores_dict.items():
        if v == 28:
            solutions += 1
    return solutions

def generate_graph(X, Y1, Y2):
    plt.xlabel("Generations")
    plt.ylabel("Score")
    plt.plot(X, Y1, color='b', label='Best score')
    plt.plot(X, Y2, color='g', label='Average score')
    plt.legend()
    plt.savefig("scores_per_generation")
