import os
import operator
from random import randint, sample


class GeneticAlgorithm:
    def __init__(self, clauses, num_literals):
        self.clauses = clauses
        self.num_clauses = len(self.clauses)
        self.num_literals = num_literals

        self.max_iterations = 100
        self.generation_size = 100
        # self.mutation_rate = 0.01
        self.selection_size = 10
        self.current_iteration = 0
        # self.crossover_p = 0.5
        # self.tournament_k = 20

        self.global_best_fitness = 0

    def is_clause_satisfied(self, valuation_list, clause):
        """
        Returns True if clause is true (satisfied) or False if not satisfied.
        """
        for literal in clause:
            if literal < 0:
                v = 1 - valuation_list[-literal - 1]
            else:
                v = valuation_list[literal - 1]
            if v == 1:
                return True
        return False


    def goal_function(self, fitness):
        if fitness == self.num_clauses:
            self.global_best_fitness = fitness


    def fitness(self, chromosome):
        """
        Number of satisfied clauses with given valuation
        """
        num_true_clauses = 0
        for c in self.clauses:
            num_true_clauses += self.is_clause_satisfied(chromosome, c)

        self.goal_function(num_true_clauses)

        return num_true_clauses


    def initial_population(self):
        solutions = [[randint(0,1) for x in range(self.num_literals)] for y in range(self.generation_size)]
        init_pop = [Chromosome(solution, self.fitness(solution)) for solution in solutions]
        return init_pop


    def stop_condition(self):
        return self.current_iteration > self.max_iterations or self.global_best_fitness != 0


    def selection(self, chromosomes):
        sorted_chromos = sorted(chromosomes, key = lambda chromo: chromo.fitness)
        selected_chromos = sorted_chromos[:10]

        return selected_chromos


    def create_generation(self, for_reproduction):
        new_generation = []

        # while len(new_generation) < self.generation_size:
        #     parents = sample(for_reproduction, 2)

    def run(self):
        chromosomes = self.initial_population()

        while not self.stop_condition():
            print('Iteration %d:' % self.current_iteration)

            for_reproduction = self.selection(chromosomes)
            print('Top solution fitness = %d' % max(chromosomes, key = lambda chromo: chromo.fitness).fitness)

            # chromosomes = self.create_generation(for_reproduction)

            self.current_iteration += 1



class Chromosome:
    def __init__(self, solution, fitness):
        self.solution = solution
        self.fitness = fitness

    def __str__(self):
        return ('f = ' + str(self.fitness) + ' %s') % [x for x in self.solution]



def clauses_from_file(filename):
    """
    Returns array of clauses [[1,2,3], [-1,2], ...] and number of literals(variables)
        """
    with open(filename, "r") as fin:
        #remove comments from beginning
        line = fin.readline()
        while(line.lstrip()[0] == 'c'):
            line = fin.readline()

        header = line.split(" ")
        num_literals = int(header[2].rstrip())

        lines = fin.readlines()

        for i in range(len(lines)):
            lines[i] = lines[i].split(" ")[:-1]
            lines[i] = [int(x) for x in lines[i]]

        return (lines, num_literals)


def main():
    clauses, num_literals = clauses_from_file(os.path.abspath("examples/aim-50-2_0-yes.cnf"))
    genetic = GeneticAlgorithm(clauses, num_literals)
    solution = genetic.run()

if __name__ == "__main__":
    main()
