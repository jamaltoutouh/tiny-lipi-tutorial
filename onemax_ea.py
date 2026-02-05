import random
import copy
import time

class OneMaxGeneticAlgorithm:
    def __init__(self, bit_length, population_size=100, mutation_rate=0.01, crossover_rate=0.8, generations=1000, elitism=True):
        """
        Initializes the OneMax Genetic Algorithm.

        :param bit_length: Length of the bitstring.
        :param population_size: Number of individuals in the population.
        :param mutation_rate: Probability of a bit flip during mutation.
        :param crossover_rate: Probability of crossover occurring between parents.
        :param generations: Maximum number of generations to run.
        :param elitism: If True, preserves the best individual from the previous generation.
        """
        self.bit_length = bit_length
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.elitism = elitism
        self.population = []
        self.best_solution = None
        self.best_fitness = 0
        self.fitness_history = []

    def initialize_population(self):
        """Generates an initial population of random solutions."""
        self.population = []
        for _ in range(self.population_size):
            # Each individual is a list of 0s and 1s
            individual = [random.choice([0, 1]) for _ in range(self.bit_length)]
            self.population.append(individual)

    def calculate_fitness(self, individual):
        """
        Calculates fitness: the number of 1s in the bitstring.
        
        :param individual: A list of 0s and 1s.
        :return: Integer sum of the list.
        """
        return sum(individual)

    def selection_tournament(self, k=3, logging=False):
        """Selects a parent using tournament selection."""
        # Pick k random individuals
        contestants = random.sample(self.population, k)
        if logging: print(f"Contestants: {contestants}")
        # Return the one with the highest fitness
        winner = max(contestants, key=lambda ind: self.calculate_fitness(ind))
        if logging: print(f"Winner: {winner}")
        return winner

    def crossover_one_point(self, parent1, parent2):
        """Performs one-point crossover."""
        if random.random() < self.crossover_rate:
            point = random.randint(1, self.bit_length - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        else:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

    def crossover_uniform(self, parent1, parent2):
        """Performs uniform crossover."""
        if random.random() < self.crossover_rate:
            child1 = []
            child2 = []
            for i in range(self.bit_length):
                if random.random() < 0.5:
                    child1.append(parent1[i])
                    child2.append(parent2[i])
                else:
                    child1.append(parent2[i])
                    child2.append(parent1[i])
            return child1, child2
        else:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

    def mutate(self, individual, logging=False):
        """Applies bit-flip mutation."""
        point = random.randint(1, self.bit_length - 1)
        individual[point] = 1 - individual[point] # Flip 0 to 1 or 1 to 0
        if logging: print(f"Mutation point:\t{point} --> Mutated:\t{individual}")
    
    def current_fitness_stats(self):
        """Calculates fitness statistics for the current population."""
        scored_population = [(ind, self.calculate_fitness(ind)) for ind in self.population]
        scored_population.sort(key=lambda x: x[1], reverse=True)
        fitness_list = [ind[1] for ind in scored_population]
        mean_fitness = sum(fitness_list) / len(fitness_list)
        std_dev_fitness = sum([(x - mean_fitness) ** 2 for x in fitness_list]) / len(fitness_list)
        std_dev_fitness = std_dev_fitness ** 0.5
        return mean_fitness, std_dev_fitness

    def run(self):
        """Runs the evolutionary loop."""
        self.initialize_population()
        
        # Evaluate initial population
        scored_population = [(ind, self.calculate_fitness(ind)) for ind in self.population]
        # Sort to find best
        scored_population.sort(key=lambda x: x[1], reverse=True)
        
        self.best_solution = scored_population[0][0]
        self.best_fitness = scored_population[0][1]
        self.fitness_history.append(self.best_fitness)

        print(f"Initial Best Fitness: {self.best_fitness}/{self.bit_length}")

        for generation in range(1, self.generations + 1):
            next_generation = []
            
            # Elitism: keep the best
            if self.elitism:
                next_generation.append(copy.deepcopy(self.best_solution))

            # Generate offspring
            while len(next_generation) < self.population_size:
                parent1 = self.selection_tournament()
                parent2 = self.selection_tournament()
                
                # Crossover
                child1, child2 = self.crossover_one_point(parent1, parent2)
                
                # Mutation
                if random.random() < self.mutation_rate: self.mutate(child1)
                if random.random() < self.mutation_rate: self.mutate(child2)
                
                next_generation.append(child1)
                if len(next_generation) < self.population_size:
                    next_generation.append(child2)
            
            self.population = next_generation
            
            # Evaluate new population
            current_best_ind = None
            current_best_fit = -1
            
            for ind in self.population:
                fit = self.calculate_fitness(ind)
                if fit > current_best_fit:
                    current_best_fit = fit
                    current_best_ind = ind
            
            # Update global best
            if current_best_fit > self.best_fitness:
                self.best_fitness = current_best_fit
                self.best_solution = copy.deepcopy(current_best_ind)
            
            self.fitness_history.append(self.best_fitness)

            # Optional: Print progress
            if generation % 100 == 0 or generation == self.generations:
                print(f"Generation {generation}: Best Fitness = {self.best_fitness}/{self.bit_length}")
            
            # Stop if perfect solution found
            if self.best_fitness == self.bit_length:
                print(f"Optimal solution found at generation {generation}!")
                break
        
        return self.best_solution, self.best_fitness


if __name__ == "__main__":
    # Example Usage
    BIT_LENGTH = 100
    POP_SIZE = 50
    GENERATIONS = 500

    print(f"Running OneMax GA for bit length {BIT_LENGTH}...")
    
    # Run GA
    ea = OneMaxGeneticAlgorithm(
        bit_length=BIT_LENGTH,
        population_size=POP_SIZE, 
        generations=GENERATIONS,
        mutation_rate=0.01,
        crossover_rate=0.8
    )
    
    start_time = time.time()
    best_sol, best_fit = ea.run()
    end_time = time.time()

    print("\n--- Results ---")
    print(f"Best Fitness: {best_fit}/{BIT_LENGTH} ({(best_fit/BIT_LENGTH)*100:.2f}%)")
    print(f"Best Solution (first 20 bits): {best_sol[:20]}...")
    print(f"Time Taken: {end_time - start_time:.4f} seconds")
