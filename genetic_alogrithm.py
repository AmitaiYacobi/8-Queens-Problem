def generate_chromosom():
    pass

def generate_population(population_size=100):
    pass

def fitness_function(population):
    pass

def selection(population):
    pass

def crossover(parent1, parent2):
    pass

def mutation(offspring1, offspring2):
    pass

def calculate_fittest_offspring(offspring1, offspring2):
    pass

def add_offspring_to_population(population, offspring):
    pass 

def run_algorithm(population):
    population = generate_population(100)
    population_scores = fitness_function(population)
    while True:
        parent1, parent2 = selection(population, population_scores)
        offspring1, offspring2 = crossover(parent1, parent2)
        offspring1, offspring2 = mutation(offspring1, offspring2)
        fittest_offspring = calculate_fittest_offspring(offspring1, offspring2)
        popultaion = add_offspring_to_population(population, fittest_offspring)
        population_scores = fitness_function(population)







if __name__ == "__main__":
    best_individual = run_algorithm()
