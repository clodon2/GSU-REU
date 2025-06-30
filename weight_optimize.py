import numpy as np
from random import sample


class DifferentialEvolution:
    def __init__(self, population_size:int, problem_dimensions:int, iterations:int,
                 scaling_factor:float, crossover_chance:float, search_space:list[float],
                 fitness_function):
        self.population_size = population_size
        self.problem_dimensions = problem_dimensions
        self.iterations = iterations
        self.scaling_factor = scaling_factor
        self.crossover_chance = crossover_chance
        self.search_space = search_space
        self.fitness_function = fitness_function

        self.population = []
        self.scores = []

    def initialize_population(self):
        for i in range(self.population_size):
            chromosome = np.random.uniform(low=self.search_space[0],
                                                     high=self.search_space[1],
                                                     size=self.problem_dimensions)
            score = self.get_fitness(chromosome)
            self.population.append(chromosome)
            self.scores.append(score)

    def iteration(self):
        for i, parent_vector in enumerate(self.population):
            print(f"calculating parent {i + 1} of {len(self.population)}")
            target_vector, solutiona, solutionb = sample(self.population, 3)
            solution_difference = solutiona - solutionb
            trial_vector = target_vector + self.scaling_factor * solution_difference
            # fit values to search space
            trial_vector = np.clip(trial_vector, self.search_space[0], self.search_space[1])

            # initialize final offspring vector
            offspring = []
            # crossover
            crossover_chances = np.random.uniform(low=0, high=1, size=self.problem_dimensions)
            for c, chance in enumerate(crossover_chances):
                if chance < self.crossover_chance:
                    offspring.append(trial_vector[c])
                else:
                    offspring.append(parent_vector[c])

            offspring = np.array(offspring)

            offspring_score = self.get_fitness(offspring)

            # if offspring gets better score, replace parent
            if offspring_score > self.scores[i]:
                self.population[i] = offspring
                self.scores[i] = offspring_score
            else:
                pass

    def get_fitness(self, chromosome):
        return self.fitness_function(chromosome)

    def run(self):
        self.initialize_population()
        for i in range(self.iterations):
            print(f"Running iteration {i + 1} of {self.iterations}")
            self.iteration()

        # sort scores for final result
        sorted_pairs = sorted(zip(self.scores, self.population),key=lambda x: x[0], reverse=True)
        # return best vector
        return sorted_pairs[:5]


if __name__ == "__main__":
    def fitness_function(vector:np.array):
        return 1 / np.sum(vector)

    DE = DifferentialEvolution(population_size=20, problem_dimensions=5, iterations=1000, scaling_factor=.5,
                               crossover_chance=.7, search_space=[-5, 5], fitness_function=fitness_function)

    print(DE.run())