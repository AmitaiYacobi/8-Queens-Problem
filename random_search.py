import matplotlib.pyplot as plt
from genetic_alogrithm import *


def generate_graph(X, Y):
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.plot(X, Y)
    plt.savefig("random_search.png")
    plt.show()

def find_solution_randomly():
    fitness = 0
    iteration = 0
    iterations = []
    fitnesses = []
    while fitness != 28:
        chromosome = generate_chromosome(chromosome_size=8)
        fitness = chromosome_fitness(chromosome)
        fitnesses.append(int(fitness))
        iterations.append(int(iteration))
        iteration += 1

    print("Found solution!")
    print(f"Solution is: {chromosome}")
    generate_graph(iterations, fitnesses)


if __name__ == "__main__":
    find_solution_randomly()