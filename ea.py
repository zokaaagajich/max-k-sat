#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from random import random, randint, uniform, sample, choice
import argparse


class EA:
    def __init__(self,
                path,
                max_iterations,
                generation_size,
                mutation_rate = None,
                reproduction_size = None,
                crossover_p = None,
                tournament_k = None,
                lambda_star = None,
                alpha = None,
                max_flip = None,
                max_table_size = None):

        #Read clauses with weights from file
        self.clauses, self.num_literals, self.num_clauses = w_clauses_from_file(os.path.abspath(path))

        self.max_iterations = max_iterations
        self.current_iteration = 1
        self.generation_size = generation_size
        self.mutation_rate = mutation_rate
        self.reproduction_size = reproduction_size
        self.crossover_p = crossover_p
        self.tournament_k = tournament_k
        #for SAWEA
        self.lambda_star = lambda_star
        #for RFEA
        self.alpha = alpha
        self.variables_weights = {i:0 for i in range(self.num_literals)}
        #for FlipGA
        self.max_flip = max_flip
        #for ASAP
        self.table = []
        self.max_table_size = max_table_size
        self.frozen = [0 for x in range(self.num_literals)]

        #Initialize population using random approach
        self.population = [[randint(0,1) for x in range(self.num_literals)] for y in range(self.generation_size)]
        self.top_chromosome = self.population[0]


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


    def fitness(self, chromosome):
        """
        Number of satisfied clauses with given valuation
        """
        num_true_clauses = 0
        for c in self.clauses:
            num_true_clauses += self.is_clause_satisfied(chromosome, c['clause'])

        return num_true_clauses


    def stop_condition(self):
        return self.current_iteration > self.max_iterations or self.fitness(self.top_chromosome) == self.num_clauses


    def selectionTop10(self):
        """
        Perform selection using top 10 best fitness chromosome approach
        """
        sorted_chromos = sorted(self.population, key = lambda chromo: self.fitness(chromo), reverse=True)
        selected_chromos = sorted_chromos[:10]

        return selected_chromos


    def selection_tournament_pick_one(self):
        """
        Chooses one chromosome using tournament selection.
        Parameter k defines how many chromosomes we take from population
        """
        #Choose k random chromosomes for battle from population and search for chromosome with
        #highest fitness
        the_chosen_ones = sample(self.population, self.tournament_k)
        winner = max(the_chosen_ones, key = lambda chromo: self.fitness_REF(chromo))
        return winner.copy()


    def selectionTournament(self):
        """
        Function chooses self.reproduction_size chromosomes using tournament selection
        """
        return [self.selection_tournament_pick_one() for i in range(self.reproduction_size)]


    def selection_roulette_pick_one(self, sum_fitness):
        """
        Chooses one chromosome using roulette selection.
        """
        pick = uniform(0, sum_fitness)
        value = 0
        i = 0

        for chromosome in self.population:
            i += 1
            value += self.fitness(chromosome)
            if value > pick:
                return chromosome.copy()


    def selectionRoulette(self):
        """
        Function chooses self.reproduction_size chromosomes using roulette selection
        """
        sum_fitness = sum(self.fitness(chromosome) for chromosome in self.population)
        return [self.selection_roulette_pick_one(sum_fitness) for i in range(self.reproduction_size)]


    def crossover(self, a, b):
        """
        Perform uniform crossover with given probability crossover_p
        """
        ab = [None]*self.num_literals
        ba = [None]*self.num_literals

        for i in range(self.num_literals):
            if random() < self.crossover_p:
                ab[i] = a[i]
                ba[i] = b[i]
            else:
                ab[i] = b[i]
                ba[i] = a[i]

        return (ab,ba)


    def mutation(self, chromosome, adaptive = False):
        """
        Performing mutation over chromosome with given probability mutation_rate
        """
        for i in range(self.num_literals):
            #In adaptive version of function don't flip frozen genes
            if adaptive and self.frozen[i]:
                continue

            if random() < self.mutation_rate:
                chromosome[i] = 1 - chromosome[i]


    def create_generation(self, for_reproduction):
        """
        With individuals obtained in for_reproduction array we generate new generation
        using genetic operators crossover and mutation. The new generation is the same
        length as at the starting point.
        """
        new_generation = []

        #While we don't fill up new_generation
        while len(new_generation) < self.generation_size:
            #Pick 2 randomly and do crossover
            parents = sample(for_reproduction, 2)
            child1, child2 = self.crossover(parents[0], parents[1])

            #Perform mutation after crossover
            self.mutation(child1)
            self.mutation(child2)

            #Add new chromosomes into new generation
            new_generation.append(child1)
            new_generation.append(child2)

        self.top_chromosome = max(new_generation, key = lambda chromo: self.fitness(chromo))
        return new_generation


    def fitness_SAW(self, chromosome):
        """
        Fitness function with weights
        in order to identify the hard clauses
        """
        return sum(map(lambda i: i['w'] * self.is_clause_satisfied(chromosome, i['clause']), self.clauses))


    def mutation_one(self, chromosome):
        """
        Mute one randomly chosen gene
        """
        i = randint(0, self.num_literals-1)
        chromosome[i] = 1 - chromosome[i]


    def lamarckian_mutation(self, chromosome):
        """
        Set of randomly chosen clauses is generated. If each clause in this set is
        satisfied by given chromosome, then do nothing. Otherwise pick a random variable of an
        unsatisfied clause and flip its corresponding bit such that it satisfies the clause.
        """
        #Generate set of randomly chosen clauses
        random_clauses = sample(self.clauses, randint(1, self.num_clauses))

        unsatisfied_clauses = self.get_unsatisfied_clauses(chromosome, random_clauses)

        for clause in unsatisfied_clauses:
            chosen_var = abs(choice(clause))
            chromosome[chosen_var-1] = 1 - chromosome[chosen_var-1]


    def create_generation_1_Lambda(self):
        """
        (1, lambda) - 1 parent reproducing lambda offspring. ',' denotes that the best individuals
        in offspring will form the next generation
        """
        parent = self.population[0]
        children = []

        #Choose lambda* offspring
        i = 0
        while i < self.lambda_star:
            child = parent.copy()
            self.mutation_one(child)
            if child not in children:
                children.append(child)
                i += 1

        #Lamarckian adaptation
        chromosome = choice(children)
        self.lamarckian_mutation(chromosome)

        self.top_chromosome = max(children, key = lambda chromo: self.fitness_SAW(chromo))
        self.population = [self.top_chromosome]


    def update_clauses_weight(self):
        """
        Update clauses weight to identify the hard clauses
        """
        for clause in self.clauses:
            clause['w'] = clause['w'] + 1 - self.is_clause_satisfied(self.top_chromosome, clause['clause'])


    def K(self, x):
        return -1 if x == 0 else 1


    def update_v(self, i):
        """
        Update weights of variables
        """
        #Summing weights of unsatisfied clauses containing variable i
        sum_weight_k = 0
        for c in self.clauses:
            if(self.is_clause_satisfied(self.top_chromosome, c['clause']) == False and ((i+1) in c['clause'] or -(i+1) in c['clause'])):
                sum_weight_k += c['w']

        self.variables_weights[i] = self.variables_weights[i] - self.K(self.top_chromosome[i]) * sum_weight_k


    def r(self, chromosome):
        """
        Function used in fitness_REF
        """
        literals = range(self.num_literals)

        arg_up = sum(map(lambda i: self.K(chromosome[i]) * self.variables_weights[i], literals))
        arg_down = 1 + sum(map(lambda i: abs(self.variables_weights[i]), literals))

        for i in literals:
            self.update_v(i)

        return 0.5*(1 + arg_up/arg_down)


    def fitness_REF(self, chromosome):
        return self.fitness(chromosome) + self.alpha * self.r(chromosome)


    def get_unsatisfied_clauses(self, chromosome, clauses):
        unsatisfied = []

        for i in clauses:
            if self.is_clause_satisfied(chromosome, i['clause']) == False:
                unsatisfied.append(i['clause'])

        return unsatisfied


    def mutation_knowledge_based(self, chromosome):
        """
        Selects an unsatisfied clause and flips exactly one randomly chosen variable contained in the clause
        """
        chosen_unsat_clause = choice(self.get_unsatisfied_clauses(chromosome, self.clauses))
        var = abs(choice(chosen_unsat_clause))
        chromosome[var-1] = 1 - chromosome[var-1]


    def create_generation_steady_state(self, for_reproduction):
        """
        Steady state replacement eliminating the worst individual
        """
        child1 = for_reproduction[0].copy()
        child2 = for_reproduction[1].copy()

        #Make offspring by mutation
        self.mutation_knowledge_based(child1)
        self.mutation_knowledge_based(child2)

        #Offspring is rejected if it is already contained in the current population
        new_in_population = 0
        if child1 not in self.population:
            self.population.append(child1)
            new_in_population += 1
        if child2 not in self.population:
            self.population.append(child2)
            new_in_population += 1

        #Eliminate worst chromosomes
        while new_in_population:
            worst = min(self.population, key = lambda chromo: self.fitness_REF(chromo))
            self.population.remove(worst)
            new_in_population -= 1

        self.top_chromosome = max(self.population, key = lambda chromo: self.fitness_REF(chromo))


    def local_search(self, chromosome, adaptive = False):
        improvement = 1
        nbrflip = 0

        while(improvement > 0 and nbrflip < self.max_flip):

            improvement = 0
            for i in range(self.num_literals):
                #In adaptive version of function don't flip frozen genes
                if adaptive and self.frozen[i]:
                    continue

                fit_before = self.fitness(chromosome)
                #Flip the i-th variable of the particle
                chromosome[i] = 1 - chromosome[i]
                nbrflip += 1
                fit_after = self.fitness(chromosome)

                gain = fit_after - fit_before
                if gain >= 0:
                    #Accept flip
                    improvement += gain
                else:
                    #There is no improvement
                    #Undo flip
                    chromosome[i] = 1 - chromosome[i]


    def two_best(self):
        first = max(self.population, key = lambda chromo: self.fitness(chromo))
        self.population.remove(first)
        second = max(self.population, key = lambda chromo: self.fitness(chromo))
        self.population.append(first)
        return [first, second]


    def create_generation_generational(self):
        new_generation = []
        new_generation_size = 0

        #Add to new generation best of current generation
        parents = self.two_best()
        new_generation.append(parents[0])
        new_generation.append(parents[1])
        new_generation_size += 2

        #While we don't fill up new_generation
        while new_generation_size < self.generation_size:

            #Select parents from population
            parents = self.selectionRoulette()

            #Apply uniform crossover
            child1, child2 = self.crossover(parents[0], parents[1])

            #Mutate each child with probability of 0.9
            if random() < 0.9:
                self.mutation(child1)
            if random() < 0.9:
                self.mutation(child2)

            #Preform Flip heuristic on each child
            self.local_search(child1)
            self.local_search(child2)

            #Add new chromosomes into new generation
            new_generation.append(child1)
            new_generation.append(child2)
            new_generation_size += 2

        self.top_chromosome = max(new_generation, key = lambda chromo: self.fitness(chromo))
        self.population = new_generation


    def num_of_equivalence_classes(self):
        """
        The chromosomes in table T are grouped into equivalence classes,
        each class containing equal chromosomes.
        """
        eq_dict = {}
        num_of_eq = 0
        for row in self.table:
            str_row = "".join(map(str, row))
            if str_row in eq_dict:
                eq_dict[str_row] += 1
            else:
                eq_dict[str_row] = 1
                num_of_eq += 1

        return num_of_eq


    def unfreeze(self):
        """
        Unfreeze all genes - set to zero
        """
        self.frozen = [0 for x in range(self.num_literals)]


    def update_table(self, child):
        """
        If Flip Heuristic directs the search towards similar local optima having equal fitness function.
        Then we can try to escape by prohibiting the flipping of some genes
        and by adapting the probability of mutation of the genes that are allowed to be modified.
        """
        parent = self.population[0]

        #Unfreeze all genes
        self.unfreeze()

        parent_fitness = self.fitness(parent)
        child_fitness = self.fitness(child)
        if parent_fitness > child_fitness:
            #Discard child
            child = parent.copy()
        elif child_fitness > parent_fitness:
            #Empty table and add child to table
            self.table.clear()
            self.table.append(child)
        else:   #child_fitness == parent_fitness
            #Add child to table
            self.table.append(child)
            #If table is full
            if len(self.table) == self.max_table_size:
                #compute frozen genes
                #NOTE:  sum of all chromosomes gene-wise in table
                #       if there is no changes across all columns - don't freeze
                #       later on, that gene can be changed with mutation or local search
                self.frozen = [(0 if x == self.max_table_size or x == 0 else 1) for x in map(sum, zip(*self.table))]

                #adapt mutation rate
                n_frozen = sum(self.frozen)
                self.mutation_rate = 0.5 * n_frozen / self.num_literals

                #count equivalence classes
                if self.num_of_equivalence_classes() <= 2:
                    #RESTART - generate new random chromosome
                    child = [randint(0,1) for x in range(self.num_literals)]
                    self.unfreeze()

                self.table.clear()

        return child


    def create_generation_1_plus_1(self):
        """
        (1+1) - 1 parent reproducing 1 child. '+' denotes elitism strategy on both generations
        """
        child = self.population[0].copy()

        self.mutation(child, adaptive = True)
        self.local_search(child, adaptive = True)
        child = self.update_table(child)

        self.top_chromosome = child
        self.population = [self.top_chromosome]



def run(path):
    ea = EA(path)

    while not ea.stop_condition():
        print('Iteration %d:' % ea.current_iteration)

        #From population choose chromosomes for reproduction

        # for_reproduction = ea.selectionTop10()
        # for_reproduction = ea.selectionTournament()
        for_reproduction = ea.selectionRoulette()

        print('Current solution fitness:\n%d' % ea.fitness(ea.top_chromosome))

        #Using genetic operators crossover and mutation create new chromosomes
        ea.population = ea.create_generation(for_reproduction)

        ea.current_iteration += 1

    #Return best chromosome in the last population
    return (ea.top_chromosome, ea.fitness(ea.top_chromosome))


def run_SAWEA(path, max_iterations, lambda_star):
    """
    Using stepwise adaption of weights
    """
    ea = EA(
        path,
        max_iterations,
        generation_size = 1,
        lambda_star = lambda_star
    )

    while not ea.stop_condition():
        print('Iteration %d:' % ea.current_iteration)

        print('Current solution fitness:\n%d' % ea.fitness(ea.top_chromosome))

        #Using genetic operators crossover and mutation create new chromosomes
        ea.create_generation_1_Lambda()

        ea.current_iteration += 1

        if ea.current_iteration % 10 == 0:
            ea.update_clauses_weight()

    return (ea.top_chromosome, ea.fitness(ea.top_chromosome), ea.current_iteration)


def run_RFEA(path, max_iterations, crossover_p, alpha):

    ea = EA(
        path,
        max_iterations,
        generation_size = 4,
        reproduction_size = 2,
        crossover_p = crossover_p,
        tournament_k = 2,
        alpha = alpha)

    while not ea.stop_condition():
        print('Iteration %d:' % ea.current_iteration)

        #From population choose chromosomes for reproduction
        for_reproduction = ea.selectionTournament()

        #print('Current solution RFEA fitness:\n%g' % ea.fitness_REF(ea.top_chromosome))
        print('Current solution fitness:\n%d' % ea.fitness(ea.top_chromosome))

        #Using genetic operators crossover and mutation create new chromosomes
        ea.create_generation_steady_state(for_reproduction)

        ea.current_iteration += 1

        if ea.current_iteration % 10 == 0:
            ea.update_clauses_weight()

    return (ea.top_chromosome, ea.fitness(ea.top_chromosome), ea.current_iteration)


def run_FlipGA(path, max_iterations, crossover_p, max_flip):

    ea = EA(
        path,
        max_iterations,
        generation_size = 10,
        reproduction_size = 2,
        mutation_rate = 0.5,
        crossover_p = crossover_p,
        max_flip = max_flip)

    #Apply flip heuristic to initial population
    for chromosome in ea.population:
        ea.local_search(chromosome)

    while not ea.stop_condition():
        print('Iteration %d:' % ea.current_iteration)

        print('Current solution fitness:\n%d' % ea.fitness(ea.top_chromosome))

        #Using genetic operators crossover and mutation create new chromosomes
        ea.create_generation_generational()

        ea.current_iteration += 1

    return (ea.top_chromosome, ea.fitness(ea.top_chromosome), ea.current_iteration)


def run_ASAP(path, max_iterations, max_flip, max_table_size):

    ea = EA(
        path,
        max_iterations,
        generation_size = 1,
        mutation_rate = 0.5,
        max_flip = max_flip,
        max_table_size = max_table_size)

    #Apply flip heuristic to parent
    ea.local_search(ea.population[0])

    while not ea.stop_condition():
        print('Iteration %d:' % ea.current_iteration)

        print('Current solution fitness:\n%d' % ea.fitness(ea.top_chromosome))

        #Using genetic operators crossover and mutation create new chromosomes
        ea.create_generation_1_plus_1()

        ea.current_iteration += 1

    return (ea.top_chromosome, ea.fitness(ea.top_chromosome), ea.current_iteration)


def w_clauses_from_file(filename):
    """
    Returns array of clauses with weights = 1
    [{'clause':[1,2,3], 'w':1}, {'clause':[-1,2], 'w':1}... ]
    and number of literals
    """
    clauses_with_weights = []

    with open(filename, "r") as fin:
        #remove comments from beginning
        line = fin.readline()
        while(line.lstrip()[0] == 'c'):
            line = fin.readline()

        header = line.split(" ")
        num_literals = int(header[2].rstrip())
        num_clauses = int(header[3].rstrip())

        lines = fin.readlines()
        for line in lines:
            line = line.split(" ")[:-1]
            line = [int(x) for x in line]
            clauses_with_weights.append({'clause':line, 'w':1})

        return (clauses_with_weights, num_literals, num_clauses)


def main():
    #Parsing arguments of command line
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help = "path to input .cnf file")
    parser.add_argument('algorithm',
                        choices=['sawea','rfea','flipga', 'asap'],
                        help = "Choose an algorithm to run")
    parser.add_argument('-i', '--max_iterations',
                        nargs = '?', default = 1000, type = int,
                        help = "Maximal number of iterations. Default 1000")
    parser.add_argument('-g', '--generation_size',
                        nargs = '?', default = 100, type = int,
                        help = "Generation size. Default 100")
    parser.add_argument('-m', '--mutation_rate',
                        nargs = '?', default = 0.01, type = float,
                        help = "Mutation probability.  Default 0.01")
    parser.add_argument('-r', '--reproduction_size',
                        nargs = '?', default = 10, type = int,
                        help = "Default 10")
    parser.add_argument('-p', '--crossover',
                        nargs = '?', default = 0.5, type = float,
                        help = "Crossover probability. Default 0.5")
    parser.add_argument('-k', '--tournament_k',
                        nargs = '?', default = 2, type = int,
                        help = "Number of elements in tournament selection. Default 2")
    parser.add_argument('-l', '--lambda_star',
                        nargs = '?', default = 10, type = int,
                        help = "lambda* for SAWEA. Default 10")
    parser.add_argument('-a', '--alpha',
                        nargs = '?', default = 0.5, type = float,
                        help = "alpha for RFEA. Default 0.5")
    parser.add_argument('-f', '--max_flip',
                        nargs = '?', default = 30000, type = int,
                        help = "Maximal number of flips for FlipGA. Default 30000")
    parser.add_argument('-t', '--max_table_size',
                        nargs = '?', default = 10, type = int,
                        help = "Maximal number of chromosomes in table. Default 10")
    args = parser.parse_args()

    #run("examples/aim-50-2_0-yes.cnf")

    if (args.algorithm == "sawea"):
        print("SAWEA".center(40, "-"))
        solution, fitness, iteration = run_SAWEA(
            path                = args.path,
            max_iterations      = args.max_iterations,
            lambda_star         = args.lambda_star
        )

    if (args.algorithm == "rfea"):
        print("RFEA".center(40, "-"))
        solution, fitness, iteration = run_RFEA(
            path                = args.path,
            max_iterations      = args.max_iterations,
            crossover_p         = args.crossover,
            alpha               = args.alpha
        )

    if (args.algorithm == "flipga"):
        print("FlipGA".center(40, "-"))
        solution, fitness, iteration = run_FlipGA(
            path                = args.path,
            max_iterations      = args.max_iterations,
            crossover_p         = args.crossover,
            max_flip            = args.max_flip
        )

    if (args.algorithm == "asap"):
        print("ASAP".center(40, "-"))
        solution, fitness, iteration = run_ASAP(
            path                = args.path,
            max_iterations      = args.max_iterations,
            max_flip            = args.max_flip,
            max_table_size      = args.max_table_size
        )


    print("Solution:")
    print(solution)
    print("Satisfied clauses:")
    print(fitness)
    print("In ", iteration, " iterations")



if __name__ == "__main__":
    main()
