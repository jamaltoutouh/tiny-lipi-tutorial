import random
import copy
import time
import matplotlib.pyplot as plt

class BinaryBilinearCCEA:
    def __init__(self, bit_length, population_size=50, mutation_rate=0.01, crossover_rate=0.8, generations=1000, elitism=True):
        """
        Initializes the Competitive Co-Evolutionary Algorithm for the Binary Bilinear Problem.
        
        Problem: max_x min_y (x^T M y)
        where x and y are binary vectors of length n.
        M is a matrix (can be Identity or random).
        
        Population X tries to MAXIMIZE the bilinear product.
        Population Y tries to MINIMIZE the bilinear product.

        :param bit_length: Length of the bitstring (and matrix dimension).
        :param population_size: Number of individuals in EACH population.
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
        
        # Two populations
        self.pop_x = []
        self.pop_y = []
        
        # Interaction Matrix M (using Identity for standard Bilinear problem, effectively Matching Pennies)
        # Or we can make it Random for a more complex landscape.
        # Let's verify the "standard" benchmark. Usually it's Identity or Upper Triangular.
        # We'll stick to Identity for simplicity unless specified, effectively maximize matching bits.
        # To make it non-trivial/zero-sum with {0,1}, let's map bits to {-1, +1} for calculation?
        # Or just use Identity on {0,1}. If x=1, y=0 -> 0. If x=1, y=1 -> 1.
        # MaxMin x^T y: x wants 1s where y has 1s. y wants 0s where x has 1s.
        # This implementation allows for any matrix logic in `calculate_bilinear_product`.
        self.matrix = None 
        
        # History
        self.history_best_x = []
        self.history_best_y = [] 

    def initialize_populations(self):
        """Generates initial populations of random solutions."""
        self.pop_x = []
        self.pop_y = []
        for _ in range(self.population_size):
            self.pop_x.append([random.choice([0, 1]) for _ in range(self.bit_length)])
            self.pop_y.append([random.choice([0, 1]) for _ in range(self.bit_length)])

    def calculate_bilinear_product(self, ind_x, ind_y):
        """
        Calculates x^T M y.
        Defaulting to Identity Matrix for the "standard" Bilinear Problem:
        Sum(x_i * y_i).
        
        If x wants to Maximize this and y Minimize:
        x tries to match y's 1s.
        y tries to mismatch x (play 0 where x is 1).
        """
        # Simple dot product (M = Identity)
        score = sum(x * y for x, y in zip(ind_x, ind_y))
        return score

    def evaluate_populations(self):
        """
        Evaluates every individual in X against every individual in Y (All-Pairs).
        Returns list of fitnesses for X and Y.
        
        Fitness_X = min_y (score(x, y))  -> Maximin objective
        Fitness_Y = min_x (-score(x, y)) -> Minimax objective (equivalent to max_x -score)
                    OR standard competitive: Score of Y is how well it minimizes.
                    Let's say Score_Y = - (x^T y). Y wants to Maximize Score_Y.
        """
        scores_x = [0] * self.population_size
        scores_y = [0] * self.population_size
        
        # All-pairs interaction (O(N^2))
        # For larger populations, use random sample.
        for i in range(self.population_size):
            outcomes_for_xi = []
            for j in range(self.population_size):
                val = self.calculate_bilinear_product(self.pop_x[i], self.pop_y[j])
                outcomes_for_xi.append(val)
                
            # X wants to maximize its worst case outcome (Maximin)
            # Or in standard Co-evolution: average score, or sum score.
            # The prompt says "Maximin Optimisation".
            scores_x[i] = min(outcomes_for_xi)
            
        # For Y, we need to see its performance against all X
        # Y wants to minimize Val. So it wants to MAXIMIZE (-Val).
        for j in range(self.population_size):
            outcomes_for_yj = []
            for i in range(self.population_size):
                val = self.calculate_bilinear_product(self.pop_x[i], self.pop_y[j])
                outcomes_for_yj.append(val)
            
            # Y wants to minimize the score. 
            # In a Maximin sense for Y (if symmetric): max_y min_x (-Val).
            # So worst case for Y is when x maximizes Val.
            # Y wants to maximize ( - (max_x Val) ).
            scores_y[j] = -max(outcomes_for_yj)

        return scores_x, scores_y

    def tournament_selection(self, population, fitnesses, k=3):
        """Selects a parent using tournament selection."""
        # Pick k random indices
        indices = random.sample(range(len(population)), k)
        # Return the one with the highest fitness
        best_idx = max(indices, key=lambda idx: fitnesses[idx])
        return population[best_idx]

    def crossover(self, parent1, parent2):
        """One-point crossover."""
        if random.random() < self.crossover_rate:
            point = random.randint(1, self.bit_length - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        else:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

    def mutate(self, individual):
        """Bit-flip mutation."""
        for i in range(self.bit_length):
            if random.random() < self.mutation_rate:
                individual[i] = 1 - individual[i]

    def evolve_population(self, population, fitnesses):
        """Applies Search Operators (Selection, Crossover, Mutation) to a population."""
        next_gen = []
        
        # Elitism
        if self.elitism:
            # Find best
            best_idx = fitnesses.index(max(fitnesses))
            next_gen.append(copy.deepcopy(population[best_idx]))
            
        while len(next_gen) < self.population_size:
            p1 = self.tournament_selection(population, fitnesses)
            p2 = self.tournament_selection(population, fitnesses)
            
            c1, c2 = self.crossover(p1, p2)
            
            self.mutate(c1)
            self.mutate(c2)
            
            next_gen.append(c1)
            if len(next_gen) < self.population_size:
                next_gen.append(c2)
                
        return next_gen

    def run(self):
        """Main Co-Evolutionary Loop."""
        self.initialize_populations()
        
        print(f"Starting Co-Evolution: {self.bit_length} bits, {self.generations} gens.")
        
        for g in range(1, self.generations + 1):
            # 1. Evaluation
            # Fitness depends on the *other* population.
            fit_x, fit_y = self.evaluate_populations()
            
            # Logging
            if g % 50 == 0 or g == 1:
                avg_x = sum(fit_x) / len(fit_x)
                avg_y = sum(fit_y) / len(fit_y)
                best_x = max(fit_x)
                best_y = max(fit_y) # Remember Y is maximizing negative score
                print(f"Gen {g}: Best X Fitness={best_x} (Avg {avg_x:.1f}) | Best Y Fitness={best_y} (Avg {avg_y:.1f})")
            
            # Store history
            self.history_best_x.append(max(fit_x))
            self.history_best_y.append(max(fit_y))
            
            # 2. Reproduction
            # Evolve X using its fitness scores
            next_pop_x = self.evolve_population(self.pop_x, fit_x)
            # Evolve Y using its fitness scores
            next_pop_y = self.evolve_population(self.pop_y, fit_y)
            
            self.pop_x = next_pop_x
            self.pop_y = next_pop_y
            
        return self.pop_x, self.pop_y

if __name__ == "__main__":
    # Settings
    N = 50 # Bit length
    
    # Run
    ccea = BinaryBilinearCCEA(bit_length=N, population_size=5, generations=50)
    final_x, final_y = ccea.run()
    
    # Final Analysis
    fit_x, fit_y = ccea.evaluate_populations()
    best_x_idx = fit_x.index(max(fit_x))
    best_x = final_x[best_x_idx]
    
    print("\n--- Final Result ---")
    print(f"Best X (Maximin Strategy): {best_x}")
    print(f"Fitness of Best X: {max(fit_x)}")
    # In simple dot product games, optimal strategy for Maximin might be specific.
    # For dot product x.y:
    # If Y plays optimal (all 0s), X gets 0 no matter what.
    # So Maximin value is 0.

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(ccea.history_best_x, label="Best X Fitness (Maximin)", color='blue')
    plt.plot(ccea.history_best_y, label="Best Y Fitness (Maximin)", color='red')
    plt.xlabel("Generation")
    plt.ylabel("Fitness Value")
    plt.title("Co-Evolution Fitness Dynamics: Binary Bilinear Problem")
    plt.legend()
    plt.grid(True)
    plt.show()

